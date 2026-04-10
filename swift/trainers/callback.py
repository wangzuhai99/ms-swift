# Copyright (c) Alibaba, Inc. and its affiliates.
import math
import os
import time

import torch
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler
from tqdm import tqdm
from transformers import TrainerCallback, trainer
from transformers.trainer_callback import DefaultFlowCallback, PrinterCallback, ProgressCallback, TrainerControl, TrainerState
from transformers.trainer_utils import IntervalStrategy, has_length
from transformers.utils import is_torch_npu_available

from swift.utils import append_to_jsonl, format_time, get_device_count, get_logger, is_mp, is_pai_training_job
from .arguments import TrainingArguments

logger = get_logger()


def get_max_cuda_memory() -> float:
    devices = list(range(get_device_count())) if is_mp() else [None]
    mems = [torch.cuda.max_memory_reserved(device=device) for device in devices]
    return sum(mems) / 1024**3


def add_train_message(logs, state, start_time) -> None:
    logs['global_step/max_steps'] = f'{state.global_step}/{state.max_steps}'
    train_percentage = state.global_step / state.max_steps if state.max_steps else 0.
    logs['percentage'] = f'{train_percentage * 100:.2f}%'
    elapsed = time.time() - start_time
    logs['elapsed_time'] = format_time(elapsed)
    if train_percentage != 0:
        logs['remaining_time'] = format_time(elapsed / train_percentage - elapsed)
    for k, v in logs.items():
        if isinstance(v, float):
            logs[k] = round(logs[k], 8)
    state.max_memory = max(getattr(state, 'max_memory', 0), get_max_cuda_memory())
    if not is_torch_npu_available():
        logs['memory(GiB)'] = round(state.max_memory, 2)

    logs['train_speed(iter/s)'] = round(state.global_step / elapsed, 6)


class ProgressCallbackNew(ProgressCallback):

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.training_bar = tqdm(desc='Train', total=state.max_steps, dynamic_ncols=True)
        self.current_step = 0
        self.start_time = time.time()

    def on_prediction_step(self, args, state: TrainerState, control, eval_dataloader=None, **kwargs):
        if state.is_world_process_zero and has_length(eval_dataloader):
            if self.prediction_bar is None:
                if self.training_bar is not None:
                    self.training_bar.fp.write('\n')
                self.prediction_bar = tqdm(
                    desc='Val', total=len(eval_dataloader), leave=True, dynamic_ncols=True, position=0)
            self.prediction_bar.update()

    def on_log(self, args: TrainingArguments, state: TrainerState, control, logs=None, **kwargs):
        add_train_message(logs, state, self.start_time)
        if not is_pai_training_job() and state.is_world_process_zero:
            jsonl_path = os.path.join(args.output_dir, 'logging.jsonl')
            append_to_jsonl(jsonl_path, logs)
        super().on_log(args, state, control, logs, **kwargs)
        if state.is_world_process_zero and self.training_bar is not None:
            self.training_bar.refresh()


class DefaultFlowCallbackNew(DefaultFlowCallback):

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        control = super().on_step_end(args, state, control, **kwargs)
        # save the last ckpt
        evaluation_strategy = args.eval_strategy if hasattr(args, 'eval_strategy') else args.evaluation_strategy
        if state.global_step == state.max_steps:
            if evaluation_strategy != IntervalStrategy.NO:
                control.should_evaluate = True
            if args.save_strategy != IntervalStrategy.NO:
                control.should_save = True
        return control

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        control = super().on_epoch_end(args, state, control, **kwargs)
        evaluation_strategy = args.eval_strategy if hasattr(args, 'eval_strategy') else args.evaluation_strategy
        if args.max_epochs is not None and args.max_epochs <= math.ceil(state.epoch):
            logger.info('Training has reached `max_epochs`. The model will be saved and the training will be exited.')
            if evaluation_strategy != IntervalStrategy.NO:
                control.should_evaluate = True
            if args.save_strategy != IntervalStrategy.NO:
                control.should_save = True
            control.should_training_stop = True
        return control


class PrinterCallbackNew(PrinterCallback):

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        return super().on_train_begin(args, state, control, **kwargs)

    def on_log(self, args, state, control, logs=None, **kwargs):
        add_train_message(logs, state, self.start_time)
        if not is_pai_training_job() and state.is_world_process_zero:
            jsonl_path = os.path.join(args.output_dir, 'logging.jsonl')
            append_to_jsonl(jsonl_path, logs)

        _ = logs.pop('total_flos', None)
        if state.is_world_process_zero:
            print(logs, flush=True)


class ProfilerCallback(TrainerCallback):
    """使用 torch.profiler 分析训练性能的回调。

    配合 TensorBoard 的 PyTorch Profiler 插件查看结果::

        pip install torch-tb-profiler
        tensorboard --logdir <output_dir>/profiler

    profiler 使用 schedule 控制采集节奏：
        wait   -> 不采集，正常训练（让 GPU 预热）
        warmup -> 开始跟踪但丢弃结果（消除探针开销）
        active -> 真正采集数据
        repeat -> 以上周期重复次数（0 = 不限制）

    ``with_stack=True`` 会记录 Python 调用栈，在 TensorBoard Trace View 中可看到
    GPU 空闲等待期间 CPU 正在执行的代码位置（如 DataLoader、collate_fn 等），
    但会带来额外开销，仅在性能分析时启用。

    ``output_dir`` 自动从 ``args.output_dir`` 推导为 ``<output_dir>/profiler``，
    无需手动指定。
    """

    def __init__(
        self,
        wait: int = 5,
        warmup: int = 3,
        active: int = 5,
        repeat: int = 1,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = True,
        with_flops: bool = True,
    ):
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_flops = with_flops
        self.profiler = None
        self._output_dir = None

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        self._output_dir = os.path.join(args.output_dir, 'profiler')
        os.makedirs(self._output_dir, exist_ok=True)

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        elif is_torch_npu_available():
            try:
                activities.append(ProfilerActivity.NPU)
            except AttributeError:
                pass

        prof_schedule = schedule(
            wait=self.wait,
            warmup=self.warmup,
            active=self.active,
            repeat=self.repeat,
        )

        self.profiler = profile(
            activities=activities,
            schedule=prof_schedule,
            on_trace_ready=tensorboard_trace_handler(self._output_dir),
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
            with_flops=self.with_flops,
        )
        self.profiler.__enter__()

        total_steps = self.wait + self.warmup + self.active
        logger.info(
            f'[Profiler] started — schedule: wait={self.wait}, warmup={self.warmup}, '
            f'active={self.active}, repeat={self.repeat} '
            f'(first trace at step {total_steps}), output_dir={self._output_dir}')

    def on_step_end(self, args, state, control, **kwargs):
        if self.profiler is not None:
            self.profiler.step()

    def on_train_end(self, args, state, control, **kwargs):
        if self.profiler is not None:
            self.profiler.__exit__(None, None, None)
            logger.info(f'[Profiler] finished — traces saved to {self._output_dir}')
            self.profiler = None


# monkey patching
trainer.DEFAULT_PROGRESS_CALLBACK = ProgressCallbackNew
trainer.DEFAULT_CALLBACKS = [DefaultFlowCallbackNew]
trainer.PrinterCallback = PrinterCallbackNew

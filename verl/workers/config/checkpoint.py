# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Backend-specific extensions of :class:`verl.trainer.config.CheckpointConfig`.

The base :class:`CheckpointConfig` lives in ``verl/trainer/config/config.py`` and
carries only fields that every backend understands (``save_contents``,
``load_contents``, ``async_save``). Anything that is meaningful only to one
training backend (e.g. mbridge options for Megatron) goes into a subclass here,
mirroring how ``ActorConfig`` / ``McoreActorConfig`` are split between
``verl/trainer/config`` and ``verl/workers/config``.
"""

from dataclasses import dataclass, field
from typing import Any

from verl.trainer.config import CheckpointConfig

__all__ = ["McoreCheckpointConfig"]


@dataclass
class McoreCheckpointConfig(CheckpointConfig):
    """Checkpoint config for the Megatron-Core backend.

    Adds the mbridge-specific knobs consumed by
    :class:`verl.utils.checkpoint.megatron_checkpoint_manager.MegatronCheckpointManager`
    when it forwards kwargs to ``bridge.save_weights()``.

    Args:
        safetensors_staging_dir (str | None): Local POSIX directory used to
            serialize each HF shard before copying it to the destination.
            ``None`` or blank (the default) writes shards directly. Set a
            directory only when that destination cannot finish safetensors
            >= 0.8 ``serialize_file`` (safetensors#764). HDFS FUSE returns
            ENOSYS from that writer's ``File::set_len`` (safetensors#787).
            mountpoint-s3 fails the sibling tempfile ``rename``/``chmod``
            with ENOSYS or EPERM (safetensors#792). Ordinary local disks stay
            unset. Staging runs only when this path is non-empty and
            safetensors is >= 0.8; older releases already write in place.
            The finished shard is still published with ``os.replace`` on the
            destination, so a mount that cannot rename at all is outside this
            workaround. This is not ``mbridge_config.distributed_filesystem``.
        mbridge_config (dict[str, Any]): Extra kwargs forwarded to
            ``bridge.save_weights``. Typical keys include
            ``distributed_filesystem`` and ``memory_efficient`` for the
            ``vanilla_mbridge`` path. Keys that are not accepted by the active
            bridge's ``save_weights`` signature are silently ignored.
    """

    safetensors_staging_dir: str | None = None
    mbridge_config: dict[str, Any] = field(default_factory=dict)

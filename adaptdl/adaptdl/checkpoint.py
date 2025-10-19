# Copyright 2020 Petuum, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
This module provides functionality to Save and load arbitrary state as part of
checkpoint-restart elasticity. The `State` class can be subclassed to define
how to save/load any state to/from persistent storage, so it can be restored
after the current job restarts and resumed from where it left off.
"""

import os
import time
import json

from adaptdl.env import checkpoint_path, replica_rank, job_id

# FIXME: Keeping global state like this will result in memory leaks for
# applications which do not restart too often.
_STATES_TO_NAMES = {}
_NAMES_TO_STATES = {}


class State(object):
    """
    This class implements An arbitrary piece of state which can be saved and
    loaded as part of a checkpoint, and synchronized across all replicas.
    Should be sub-classed to define custom save, load, and sync logic.
    """

    def __init__(self, name):
        """
        Initialize the state object with a unique identifier `name`, which is
        used to refer to the saved object in persistent storage. No two `State`
        objects may share the same `name`.

        Arguments:
            name (str): Unique name of this `State` object.

        Raises:
            ValueError: If a `State` object with the given name already exists.
        """
        if name in _NAMES_TO_STATES:
            raise ValueError("State '{}' already exists".format(name))
        _NAMES_TO_STATES[name] = self
        _STATES_TO_NAMES[self] = name

    def save(self, fileobj):
        """
        This method should be overridden by subclasses to define how the state
        is saved. Is invoked by `save_all_states` and `save_state` to save the
        state into persistent storage.

        Arguments:
            fileobj (BinaryIO): A binary writable file object.
        """
        pass

    def load(self, fileobj):
        """
        This method should be overridden by subclasses to define how the state
        is loaded. Is invoked by `load_state` to load the state from persistent
        storage.

        Arguments:
            fileobj (BinaryIO): A binary readable file object.
        """
        pass

    def sync(self):
        """
        This method should be overridden by subclasses to define how the state
        is synchronized across replicas. This might be necessary to make sure
        the state is consistent before saving it to persistent storage. Is
        invoked by `save_state` before saving the state.
        """
        pass


def save_all_states():
    """
    Invokes `save_state` on all `State` objects for which `State.skip` is True.
    This function can be used to trigger a global checkpoint and save every
    `State` in the current job.

    Saves checkpoints in order of size (largest first) to minimize lost work
    if pod is terminated during checkpoint save. Model checkpoint (dataparallel)
    is saved first as it's the largest and most expensive to recompute.
    """
    all_states = list(_STATES_TO_NAMES.keys())

    # Separate states by type - save model (largest) first
    model_states = [s for s in all_states if 'dataparallel' in _STATES_TO_NAMES[s]]
    other_states = [s for s in all_states if s not in model_states]

    # Save model checkpoints first (largest, most expensive)
    for state in model_states:
        save_state(state)

    # Then save other checkpoints
    for state in other_states:
        save_state(state)


def save_state(state, sync=True):
    """
    Saves a `State` object to persistent storage. First invokes `State.sync` on
    all replicas if `sync` is `True` (default), and then invokes `State.save`
    on the replica of rank 0 only.

    Uses atomic write pattern: writes to temp file, fsyncs, then atomically
    renames to final destination to prevent corruption on pod termination.

    Arguments:
        state (State): The `State` object to save to persistent storage.
        sync (bool): Whether `State.sync` should be invoked.
    """
    if sync:
        state.sync()
    if replica_rank() == 0:
        name = _STATES_TO_NAMES[state]
        if checkpoint_path() is not None:
            final_path = os.path.join(checkpoint_path(), name)
            temp_path = final_path + ".tmp"

            overall_start = time.time()
            print(f"\n[CHECKPOINT] Starting save for '{name}'")
            print(f"[TIMING] Save started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(overall_start))} ({overall_start:.3f})")

            # Phase 1: Write to temporary file
            write_start = time.time()
            with open(temp_path, "wb") as f:
                state.save(f)
                f.flush()  # Flush Python buffers to OS
                write_end = time.time()

                # Phase 2: fsync to ensure data reaches EFS/NFS
                print(f"[TIMING] '{name}' write to buffer completed: {write_end - write_start:.3f}s")
                fsync_start = time.time()
                os.fsync(f.fileno())
                fsync_end = time.time()
                print(f"[TIMING] '{name}' fsync completed: {fsync_end - fsync_start:.3f}s")

            # Phase 3: Atomic rename
            rename_start = time.time()
            os.rename(temp_path, final_path)
            rename_end = time.time()
            print(f"[TIMING] '{name}' atomic rename completed: {rename_end - rename_start:.3f}s")

            # Calculate and log total duration
            overall_end = time.time()
            total_duration = overall_end - overall_start
            write_duration = write_end - write_start
            fsync_duration = fsync_end - fsync_start
            rename_duration = rename_end - rename_start

            file_size_mb = os.path.getsize(final_path) / (1024 * 1024)
            print(f"[CHECKPOINT] '{name}' save completed successfully")
            print(f"[TIMING] Total: {total_duration:.3f}s | Write: {write_duration:.3f}s | "
                  f"Fsync: {fsync_duration:.3f}s | Rename: {rename_duration:.3f}s | "
                  f"Size: {file_size_mb:.2f}MB")
            print(f"[TIMING] Completed at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(overall_end))} ({overall_end:.3f})\n")

            # Log to file for analysis
            log_path = os.path.join(checkpoint_path(), ".adaptdl-checkpoint-times.log")
            record = {
                "state_name": name,
                "total_duration_s": round(total_duration, 3),
                "write_duration_s": round(write_duration, 3),
                "fsync_duration_s": round(fsync_duration, 3),
                "rename_duration_s": round(rename_duration, 3),
                "file_size_mb": round(file_size_mb, 2),
                "timestamp": overall_end
            }
            with open(log_path, "a") as lf:
                lf.write(json.dumps(record) + "\n")



def load_state(state):
    """
    Load the given `State` object from persistent storage. If the object was
    previously saved, then State.load will be invoked with a readable file
    object to load from.

    Implements retry logic to handle transient NFS/EFS issues where checkpoint
    file may exist but data is not yet fully available due to async writeback.

    Arguments:
        state (State): `State` object to load from persistent storage.

    Returns:
        `True` if state was previously saved and `State.load` was invoked,
        `False` otherwise.
    """
    if checkpoint_path() is None:
        return False

    name = _STATES_TO_NAMES[state]
    checkpoint_file = os.path.join(checkpoint_path(), name)

    # Check if file doesn't exist - no need to retry
    if not os.path.exists(checkpoint_file):
        print(f"[CHECKPOINT] No checkpoint found for '{name}' at {checkpoint_file}")
        return False

    # File exists, try loading with retry logic
    max_retries = 50
    retry_delay = 10  # seconds

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"[CHECKPOINT] Retry {attempt}/{max_retries} loading '{name}' after {retry_delay}s delay...")
                time.sleep(retry_delay)

            print(f"[CHECKPOINT] Loading '{name}' from {checkpoint_file} (attempt {attempt + 1}/{max_retries})")
            load_start = time.time()

            with open(checkpoint_file, "rb") as f:
                state.load(f)

            load_end = time.time()
            load_duration = load_end - load_start

            print(f"[CHECKPOINT] Successfully loaded '{name}' in {load_duration:.3f}s")
            if attempt > 0:
                print(f"[CHECKPOINT] SUCCESS after {attempt} retries!")

            return True

        except FileNotFoundError:
            # File disappeared between existence check and open
            print(f"[CHECKPOINT ERROR] File '{name}' disappeared during load attempt {attempt + 1}")
            if attempt == max_retries - 1:
                print(f"[CHECKPOINT ERROR] Max retries reached, file not found")
                return False

        except (EOFError, OSError, ValueError, RuntimeError) as e:
            # Checkpoint corruption or incomplete write
            error_type = type(e).__name__
            print(f"[CHECKPOINT ERROR] Failed to load '{name}' (attempt {attempt + 1}/{max_retries}): {error_type}: {str(e)}")

            # Check file size for debugging
            try:
                file_size = os.path.getsize(checkpoint_file)
                print(f"[CHECKPOINT DEBUG] File size: {file_size / (1024 * 1024):.2f} MB")
            except:
                pass

            if attempt == max_retries - 1:
                print(f"[CHECKPOINT ERROR] Max retries ({max_retries}) reached, giving up")
                raise RuntimeError(f"Failed to load checkpoint '{name}' after {max_retries} attempts. "
                                 f"Last error: {error_type}: {str(e)}")

        except Exception as e:
            # Unexpected error - log and re-raise
            error_type = type(e).__name__
            print(f"[CHECKPOINT ERROR] Unexpected error loading '{name}': {error_type}: {str(e)}")
            if attempt == max_retries - 1:
                raise

    # Should not reach here
    return False

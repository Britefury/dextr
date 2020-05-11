# The MIT License (MIT)
#
# Copyright (c) 2020 University of East Anglia, Norwich, UK
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# Developed by Geoffrey French in collaboration with Dr. M. Fisher and
# Dr. M. Mackiewicz.

import os, sys
import pathlib
import torch

class LogAlreadyExistsError(Exception):
    pass


class _LogHelper(object):
    def __init__(self, path, stream):
        self.path = path
        self.stream = stream

    def write(self, x):
        with self.path.open('a+') as f_out:
            f_out.write(x)
        self.stream.write(x)

    def flush(self):
        self.stream.flush()


class JobOutput(object):
    def __init__(self, job_name, continue_job):
        log_dir = pathlib.Path('logs')
        checkpoints_dir = 'checkpoints'

        if not log_dir.exists():
            log_dir.mkdir(parents=True)

        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir, exist_ok=True)

        self.log_path = log_dir / 'log_{}.txt'.format(job_name)
        self.checkpoint_path = os.path.join(checkpoints_dir, '{}.pth'.format(job_name))

        if not continue_job and self.log_path.exists():
            raise ValueError('Log file {} already exists.'.format(str(self.log_path)))

        if self.log_path is not None:
            self.__stdout = _LogHelper(self.log_path, sys.stdout)
            self.__stderr = _LogHelper(self.log_path, sys.stderr)

    def connect_streams(self):
        if self.log_path is not None:
            sys.stdout = self.__stdout
            sys.stderr = self.__stderr

    def disconnect_streams(self):
        if self.log_path is not None:
            sys.stdout = self.__stdout.stream
            sys.stderr = self.__stderr.stream

    def read_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            return torch.load(self.checkpoint_path)
        else:
            return None

    def write_checkpoint(self, data):
        # Write to a temporary new path, in case the job gets terminated during writing
        new_path = self.checkpoint_path + '.new'
        with open(new_path, 'wb') as f_ckpt:
            torch.save(data, f_ckpt)
        # Remove the old one if it exists
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)
        # Rename
        os.rename(new_path, self.checkpoint_path)


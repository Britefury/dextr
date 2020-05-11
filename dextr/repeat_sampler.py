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

import itertools
from torch.utils.data import Sampler

class RepeatSampler(Sampler):
    r"""Repeated sampler

    Arguments:
        data_source (Dataset): dataset to sample from
        sampler (Sampler): sampler to draw from repeatedly
        repeats (int): number of repetitions or -1 for infinite
    """

    def __init__(self, sampler, repeats=-1):
        if repeats < 1 and repeats != -1:
            raise ValueError('repeats should be positive or -1')
        self.sampler = sampler
        self.repeats = repeats

    def __iter__(self):
        if self.repeats == -1:
            reps = itertools.repeat(self.sampler)
            return itertools.chain.from_iterable(reps)
        else:
            reps = itertools.repeat(self.sampler, self.repeats)
            return itertools.chain.from_iterable(reps)

    def __len__(self):
        if self.repeats == -1:
            return 2 ** 62
        else:
            return len(self.sampler) * self.repeats

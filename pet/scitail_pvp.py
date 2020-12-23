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

"""
To add a new task to PET, both a DataProcessor and a PVP for this task must
be added. The PVP is responsible for applying patterns to inputs and mapping
labels to their verbalizations (see the paper for more details on PVPs).
This file shows an example of a PVP for a new task.
"""

from typing import List

from pet.pvp import PVP, PVPS
from pet.utils import InputExample


class SciTailPVP(PVP):

    # Set this to the name of the task
    TASK_NAME = "scitail"

    # Set this to the verbalizer for the given task: a mapping from the task's labels (which can be obtained using
    # the corresponding DataProcessor's get_labels method) to tokens from the language model's vocabulary
    VERBALIZER = {
        "entails": ['yes', 'so', 'and'],
        "neutral": ['no', 'but']
    }

    def get_parts(self, example: InputExample):
        """
        This function defines the actual patterns: It takes as input an example and outputs the result of applying a
        pattern to it. To allow for multiple patterns, a pattern_id can be passed to the PVP's constructor. This
        method must implement the application of all patterns.
        """

        # We tell the tokenizer that both text_a and text_b can be truncated if the resulting sequence is longer than
        # our language model's max sequence length.
        text_a = self.shortenable(self.lowercase_first(example.text_a))
        text_b = self.shortenable(self.remove_final_punc(example.text_b))

        return ['"', text_b, '" ?'], [self.mask, ', "', text_a, '"']

    def verbalize(self, label) -> List[str]:
        return SciTailPVP.VERBALIZER[label]


# register the PVP for this task with its name
PVPS[SciTailPVP.TASK_NAME] = SciTailPVP

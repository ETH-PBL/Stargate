#
# Copyright (C) 2022-2024 ETH Zurich
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# SPDX-License-Identifier: GPL-3.0
# ======================================================================
#
# Authors: 
# Konstantin Kalenberg, ETH Zurich (kkalenbe@ethz.ch)
# Hanna Müller ETH Zurich (hanmuell@iis.ee.ethz.ch)
# Tommaso Polonelli, ETH Zurich (tommaso.polonelli@pbl.ee.ethz.ch)
# Alberto Schiaffino, ETH Zurich (aschiaffino@ethz.ch)
# Vlad Niculescu, ETH Zurich (vladn@ethz.ch)
# Cristian Cioflan, ETH Zurich (cioflanc@ethz.ch)
# Michele Magno, ETH Zurich (michele.magno@pbl.ee.ethz.ch)
# Luca Benini, ETH Zurich (lbenini@iis.ee.ethz.ch)
#


from nntool.api import NNGraph

from training_gate_navigator_TensorFlow import training_gate_navigator
from training_gate_classifier_PyTorch import training_gate_classifier
from nntool_gate_navigator_Tensorflow import validation_score_quantized_nav_model
from nntool_gate_classifier_PyTorch import quantize_classifier

if __name__ == "__main__":
 
    print("\n"+"#"*30)
    print("GATE_NAVIGATOR TRAINING")
    print("#"*30+"\n")
    
    training_gate_navigator()    
    validation_score_quantized_nav_model()

    print("\n"+"#"*30)
    print("GATE_CLASSIFIER TRAINING")
    print("#"*30+"\n")
    
    training_gate_classifier()
    quantize_classifier()


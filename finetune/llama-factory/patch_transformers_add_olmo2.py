import transformers
import os

#print transformers version
print("Transformers version:")
print(transformers.__version__)

from transformers.models.auto import configuration_auto

#need to move olmo2 folder from llama-factory/src/llamafactory/olmo2 to llama-factory/olmo2
from olmo2.configuration_olmo2 import Olmo2Config 
from olmo2.modeling_olmo2 import Olmo2ForCausalLM

#add olmo2 to CONFIG_MAPPING
configuration_auto.CONFIG_MAPPING["olmo2"] = Olmo2Config


# root = os.path.dirname(transformers.__file__)
# models_dir = os.path.join(root, "models")

# print("Transformers models directory:")
# print(models_dir)

# print("Models in transformers:")
# print(os.listdir(models_dir))

print("Complete!")
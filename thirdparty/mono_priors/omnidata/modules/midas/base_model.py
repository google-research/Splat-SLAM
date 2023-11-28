# Copyright 2024 The Omnidata-Tool Authors.

# Licensed under the License issued by the Omnidata Authors
# available here: https://github.com/EPFL-VILAB/omnidata/blob/main/LICENSE

import torch


class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device('cpu'))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)

'''
This file contains the model class for the multi-label classification model.
'''

import torch

import transformers
from transformers import (
    AutoModel,
)

class TokenMultiClassificationModel(transformers.PreTrainedModel):
    '''
    A multi-label classification model based on a pretrained transformer model.
    '''
    def __init__(self, config, pretrained_model_name_or_path, num_attribute_names):
        super().__init__(config)
        self.config = config
        self.num_attribute_names = num_attribute_names
        self.num_labels = config.num_labels
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )

        self.encoder = AutoModel.from_pretrained(
            pretrained_model_name_or_path = pretrained_model_name_or_path,
            config = config,
            add_pooling_layer=False,
        )
        self.dropout = torch.nn.Dropout(classifier_dropout)
        self.classifier = torch.nn.Linear(
            config.hidden_size,
            self.num_labels * self.num_attribute_names
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        token_type_ids = None,
        tag_ids = None,
        **kwargs,
    ):
        batch_size = input_ids.shape[0]
        seq_length = input_ids.shape[1]

        outputs = self.encoder(
            input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            **kwargs
        )

        sequence_output = outputs[0] # [B, L, H]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output) # [B, L, C*A]
        logits = logits.reshape(
            batch_size, seq_length, self.num_attribute_names, self.num_labels
        ) # [B, L, A, C]

        loss = None
        if tag_ids is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), tag_ids.view(-1))

        return transformers.modeling_outputs.MultipleChoiceModelOutput(
            loss=loss,
            logits=logits,
        )

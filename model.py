'''
This file contains the model class for the multi-label classification model.
'''

import torch

import transformers
from transformers import (
    AutoModel,
)

from torchcrf import CRF

from logzero import logger

class TokenMultiClassificationModel(transformers.PreTrainedModel):
    '''
    A multi-label classification model based on a pretrained transformer model.
    '''
    def __init__(self,
        config,
        pretrained_model_name_or_path,
        num_attribute_names,
        use_crf = False,
    ):
        super().__init__(config)
        self.config = config
        self.num_attribute_names = num_attribute_names
        self.num_labels = config.num_labels
        self.use_crf = use_crf
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

        if use_crf:
            self.crf = CRF(self.num_labels, batch_first=True)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        token_type_ids = None,
        tag_ids = None,
        decode = False,
        **kwargs,
    ):
        batch_size = input_ids.shape[0]
        seq_length = input_ids.shape[1]
        num_labels = self.num_labels
        num_attribute_names = self.num_attribute_names

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
        decoded = None
            
        if tag_ids is not None:
            if self.use_crf:
                mask = tag_ids != -100
                mask[:, 0, :] = True # mask of the first time-step must all be on
                tag_ids_no_negative = torch.where(tag_ids >= 0, tag_ids, 0)
                if not decode:
                    loss = -self.crf(
                        logits.transpose(1, 2).reshape(-1, seq_length, num_labels), # [B*A, L, C]
                        tag_ids_no_negative.transpose(1, 2).reshape(-1, seq_length), # [B*A, L]
                        mask=mask.transpose(1, 2).reshape(-1, seq_length), # [B*A, L]
                        reduction='mean'
                    )
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), tag_ids.view(-1))

        if decode:
            if self.use_crf:
                decoded = self.crf.decode(
                    logits.transpose(1, 2).reshape(-1, seq_length, num_labels), # [B*A, L, C]
                    mask=mask.transpose(1, 2).reshape(-1, seq_length), # [B*A, L]
                )
                decoded = torch.stack([
                    torch.nn.ConstantPad1d([0, seq_length - len(tags)], -100)(
                        torch.tensor(tags)
                    )
                    for tags in decoded
                ])
                # [B * A, L] -> [B, A, L]
                decoded = decoded.reshape(batch_size, num_attribute_names, seq_length)
                decoded = decoded.transpose(1, 2) # [B, L, A]
            else:
                decoded = logits.argmax(dim=-1)

        #return transformers.modeling_outputs.MultipleChoiceModelOutput(
        #    loss=loss,
        #    logits=logits,
        #)
        return dict(
            loss=loss,
            logits=logits,
            decoded=decoded,
        )

    #def decode(self, logits):
    #    if self.use_crf:
    #        return self.crf.decode(logits)
    #    else:
    #        #return torch.argmax(logits, dim=-1)
    #        return logits.argmax(dim=-1)
        
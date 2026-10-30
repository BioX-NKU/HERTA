import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
class ATACBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, num_groups=32):
        super(ATACBasicBlock, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=num_groups,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class resblock(nn.Module):
    def __init__(self, ni):
        super(resblock, self).__init__()
        self.blocks = nn.Sequential(
            nn.Conv1d(ni, ni, 3, 1, 1),
            nn.BatchNorm1d(ni),
            nn.ReLU(),
            nn.Conv1d(ni, ni, 3, 1, 1),
            nn.BatchNorm1d(ni),
            nn.ReLU(),
        )

    def forward(self, x):
        residual = x
        out = self.blocks(x)
        out = out + residual
        return out

class symmetrize_bulk(nn.Module):
    def __init__(self):
        super(symmetrize_bulk, self).__init__()

    def forward(self, x):
        if len(x.shape) == 2:
            print("not implemented")
            return None
        else:
            if len(x.shape) == 3:
                a, b, c = x.shape
                x = x.reshape(a, b, 1, c)
                x = x.repeat(1, 1, c, 1)
                x_t = x.permute(0, 1, 3, 2)
                x_sym = torch.concat((x, x_t), axis=1)
                return x_sym
            else:
                return None

class Enhanced_Branch_Pbulk(nn.Module):

    
    def __init__(self):
        super(Enhanced_Branch_Pbulk, self).__init__()
        
       
        self.feature_enhancer = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(),
            ATACBasicBlock(32, 64, num_groups=16, stride=1),
            ATACBasicBlock(64, 64, num_groups=16, stride=1),
            nn.Dropout(0.05),
            ATACBasicBlock(64, 96, num_groups=24, stride=1),
            nn.Conv1d(96, 8, kernel_size=1, bias=False),
            nn.BatchNorm1d(8), nn.ReLU(),
        )
        
        pbulk_res = 50
        
        self.bulk_summed_2d = nn.Sequential(
            nn.AvgPool1d(kernel_size=np.int64(1e04 / pbulk_res)), 
            symmetrize_bulk()
        )

        self.bulk_extractor_2d = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=11, stride=1, dilation=1, padding="same"),
            nn.BatchNorm1d(16), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=36, kernel_size=7, stride=1, dilation=1, padding="same"),
            nn.BatchNorm1d(36), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=36, out_channels=36, kernel_size=5, stride=1, dilation=1, padding="same"),
            nn.BatchNorm1d(36), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=36, out_channels=36, kernel_size=5, stride=1, dilation=1, padding="same"),
            nn.BatchNorm1d(36), nn.ReLU(),
            nn.Conv1d(in_channels=36, out_channels=36, kernel_size=5, stride=1, dilation=1, padding="same"),
            nn.BatchNorm1d(36), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels=36, out_channels=36, kernel_size=5, stride=1, dilation=2, padding="same"),
            nn.BatchNorm1d(36), nn.ReLU(),
            nn.Conv1d(in_channels=36, out_channels=36, kernel_size=5, stride=1, dilation=3, padding="same"),
            nn.BatchNorm1d(36), nn.ReLU(),
            nn.Conv1d(in_channels=36, out_channels=36, kernel_size=5, stride=1, dilation=5, padding="same"),
            nn.BatchNorm1d(36), nn.ReLU(),
            nn.Conv1d(in_channels=36, out_channels=36, kernel_size=5, stride=1, dilation=5, padding="same"),
            nn.BatchNorm1d(36), nn.ReLU(),
            nn.Conv1d(in_channels=36, out_channels=36, kernel_size=5, stride=1, dilation=7, padding="same"),
            nn.BatchNorm1d(36), nn.ReLU(),
            nn.Conv1d(in_channels=36, out_channels=36, kernel_size=5, stride=1, dilation=11, padding="same"),
            nn.BatchNorm1d(36), nn.ReLU(),
            nn.Conv1d(in_channels=36, out_channels=36, kernel_size=5, stride=1, dilation=11, padding="same"),
            nn.BatchNorm1d(36), nn.ReLU(),
            nn.MaxPool1d(kernel_size=5),
            nn.Conv1d(in_channels=36, out_channels=16, kernel_size=3, stride=1, dilation=1, padding="same"),
            nn.BatchNorm1d(16), nn.ReLU(),
            nn.MaxPool1d(kernel_size=5),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, dilation=1, padding="same"),
            nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, dilation=1, padding="same"),
            nn.BatchNorm1d(16), nn.ReLU(),
            symmetrize_bulk(),
        )

        self.total_extractor_2d = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2),
            nn.BatchNorm2d(16), nn.ReLU(),
        )


        self.side_fine = nn.Sequential(
            nn.Conv1d(8, 32, kernel_size=7, stride=4, padding=3),  # -> [B, 32, 20050]
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=4, padding=2),  # -> [B, 64, 5012]
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=4, padding=2),  # -> [B, 128, 1253]
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),  # -> [B, 256, 627]
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(256),  # -> [B, 256, 256]
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512), nn.ReLU(),
            nn.AdaptiveAvgPool1d(64),  # -> [B, 512, 64]
            nn.Flatten(),
            nn.Dropout(0.05),
            nn.Linear(512 * 64, 512),
        )
        
        # Medium track
        self.side_medium = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=5, stride=2, padding=2),  # spatial / 2
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 96, kernel_size=5, stride=2, padding=2),  # spatial / 2
            nn.BatchNorm2d(96), nn.ReLU(),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),  # spatial / 2
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1),  # spatial / 2
            nn.BatchNorm2d(192), nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),  # -> [B, 192, 8, 8]
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # -> [B, 256, 4, 4]
            nn.Flatten(),
            nn.Dropout(0.08),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(1024, 512),
        )
        
        # Coarse track: 
        self.side_coarse = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # -> [B, 64, 4, 4]
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(64 * 4 * 4, 512),
        )

    def forward(self, x2):
       
        enhanced_features = self.feature_enhancer(x2)  # [B, 8, 80200]
        
        x3_2d = self.bulk_summed_2d(enhanced_features)
        x2_2d = self.bulk_extractor_2d(enhanced_features)
        x_2d_concat = torch.cat((x3_2d, x2_2d), 1)  # [B, 48, H, W]
        
        x4 = self.total_extractor_2d(x_2d_concat)  # [B, 16, h, w]
        
       
        track_fine = self.side_fine(enhanced_features)      # [B, 512]
        track_medium = self.side_medium(x_2d_concat)        # [B, 512]
        track_coarse = self.side_coarse(x4)                 # [B, 512]
        
        return {
            'fine': track_fine,
            'medium': track_medium,
            'coarse': track_coarse
        }

class branch_topic(nn.Module):
    """
    Multi-track topic branch with convolutional side outputs.
    """
    def __init__(self):
        super(branch_topic, self).__init__()

        # Original feature-extraction flow is preserved.
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=50, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            resblock(64),
            nn.MaxPool1d(kernel_size=2),
            resblock(64),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
        )

        # Three side-output tracks.
        # Fine track
        self.side_fine = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(256),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(128),
            nn.Flatten(),
            nn.Linear(256 * 128, 512),
        )
        
        # Medium track
        self.side_medium = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(128),
            nn.Flatten(),
            nn.Linear(128 * 128, 512),
        )
        
        # Coarse track
        self.side_coarse = nn.Sequential(
            nn.AdaptiveAvgPool1d(128),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(64),
            nn.Flatten(),
            nn.Linear(128 * 64, 512),
        )

    def forward(self, x):
        # Transpose topic input for Conv1d.
        # Input: [B, 8020, 50]
        # Conv1d expects: [B, 50, 8020]
        x = x.transpose(1, 2)  # [B, 50, 8020]
        
        # Preserved feature-extraction flow.
        x = self.feature_extractor(x)  # [B, 64, L]
        
        # Side outputs.
        track_fine = self.side_fine(x)
        track_medium = self.side_medium(x)
        track_coarse = self.side_coarse(x)
        
        return {
            'fine': track_fine,
            'medium': track_medium,
            'coarse': track_coarse
        }

class SingleResolutionHead(nn.Module):
    """Per-resolution V-stripe prediction head."""
    def __init__(self, in_features, target_length, vstripe_length, hidden=128):
        super(SingleResolutionHead, self).__init__()
        
        self.target_length = target_length
        self.vstripe_length = vstripe_length
        
        initial_length = min(32, target_length // 8)
        self.feature_to_sequence = nn.Sequential(
            nn.Linear(in_features, hidden * initial_length),
            nn.ReLU()
        )
        self.initial_length = initial_length
        
        self.dilated_processor = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=5, dilation=1, padding=2),
            nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, dilation=2, padding=4),
            nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, dilation=3, padding=6),
            nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, dilation=5, padding=10),
            nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, dilation=1, padding=1),
            nn.BatchNorm1d(hidden), nn.ReLU(),
        )
        
        self.upsampler = self._build_upsampler(initial_length, target_length, hidden)
        
        self.final_conv = nn.Sequential(
            nn.Conv1d(hidden, hidden//2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden//2), nn.ReLU(),
            nn.Conv1d(hidden//2, 1, kernel_size=1),
            nn.Flatten()
        )
    
    def _build_upsampler(self, initial_length, target_length, hidden):
        layers = []
        current_length = initial_length
        
        while current_length < target_length:
            if current_length * 4 <= target_length:
                layers.extend([
                    nn.ConvTranspose1d(hidden, hidden, kernel_size=4, stride=4, padding=0),
                    nn.BatchNorm1d(hidden), nn.ReLU()
                ])
                current_length *= 4
            elif current_length * 2 <= target_length:
                layers.extend([
                    nn.ConvTranspose1d(hidden, hidden, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm1d(hidden), nn.ReLU()
                ])
                current_length *= 2
            else:
                break
        
        if current_length != target_length:
            layers.append(nn.AdaptiveAvgPool1d(target_length))
        
        return nn.Sequential(*layers)
    
    def forward(self, input_features):
        batch_size = input_features.size(0)
        x = self.feature_to_sequence(input_features)
        x = x.view(batch_size, -1, self.initial_length)
        x_processed = self.dilated_processor(x)
        x_upsampled = self.upsampler(x_processed)
        vstripe_output = self.final_conv(x_upsampled)
        return vstripe_output

class MultiScaleFusionDecoder_6Track(nn.Module):
    """6-track fusion decoder for two 3-track branches.

    In the final HERTA model, the two branches are Pbulk+CTCF and Topic.
    """
    def __init__(self, single_track_dim=512, output_dim=1536, hidden=256, num_groups=16):
        super(MultiScaleFusionDecoder_6Track, self).__init__()
        
        self.single_track_dim = single_track_dim
        total_dim = single_track_dim * 6  # Six tracks only.
        self.output_dim = output_dim
        
        self.cross_track_mixer = nn.Sequential(
            nn.Conv1d(total_dim, total_dim, kernel_size=1, groups=num_groups, bias=False),
            nn.BatchNorm1d(total_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(total_dim, total_dim, kernel_size=1, groups=num_groups, bias=False),
            nn.BatchNorm1d(total_dim),
            nn.ReLU(),
        )
        
        self.prepare_5kb = nn.Sequential(
            nn.Conv1d(total_dim, output_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )
        
        self.prepare_10kb = nn.Sequential(
            nn.Conv1d(total_dim, output_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )
        
        self.prepare_25kb = nn.Sequential(
            nn.Conv1d(total_dim, output_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )
        
        self.head_25kb = SingleResolutionHead(output_dim, 80, 80, hidden)
        self.head_10kb = SingleResolutionHead(output_dim, 200, 200, hidden)
        self.head_5kb = SingleResolutionHead(output_dim, 400, 400, hidden)
        
    def forward(self, pbulk_tracks, topic_tracks):
        # Concatenate six tracks: three Pbulk tracks and three topic tracks.
        all_tracks = torch.cat([
            pbulk_tracks['fine'], pbulk_tracks['medium'], pbulk_tracks['coarse'],
            topic_tracks['fine'], topic_tracks['medium'], topic_tracks['coarse'],
        ], dim=1)
        
        all_tracks = all_tracks.unsqueeze(2)
        mixed_tracks = self.cross_track_mixer(all_tracks)
        
        feat_5kb = self.prepare_5kb(mixed_tracks).squeeze(2)
        feat_10kb = self.prepare_10kb(mixed_tracks).squeeze(2)
        feat_25kb = self.prepare_25kb(mixed_tracks).squeeze(2)
        
        predictions = {
            '25kb': self.head_25kb(feat_25kb),
            '10kb': self.head_10kb(feat_10kb),
            '5kb': self.head_5kb(feat_5kb)
        }
        
        return predictions

class BulkOnlyMultiScaleFusionDecoder(nn.Module):
    """3-track fusion decoder for stage-1 Pbulk+CTCF pretraining."""
    def __init__(self, single_track_dim=512, output_dim=1024, hidden=256, num_groups=16):
        super(BulkOnlyMultiScaleFusionDecoder, self).__init__()
        
        self.single_track_dim = single_track_dim
        total_dim = single_track_dim * 3
        self.output_dim = output_dim
        
        self.cross_track_mixer = nn.Sequential(
            nn.Conv1d(total_dim, total_dim, kernel_size=1, groups=num_groups, bias=False),
            nn.BatchNorm1d(total_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(total_dim, total_dim, kernel_size=1, groups=num_groups, bias=False),
            nn.BatchNorm1d(total_dim),
            nn.ReLU(),
        )
        
        self.prepare_5kb = nn.Sequential(
            nn.Conv1d(total_dim, output_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )
        
        self.prepare_10kb = nn.Sequential(
            nn.Conv1d(total_dim, output_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )
        
        self.prepare_25kb = nn.Sequential(
            nn.Conv1d(total_dim, output_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )
        
        self.head_25kb = SingleResolutionHead(output_dim, 80, 80, hidden)
        self.head_10kb = SingleResolutionHead(output_dim, 200, 200, hidden)
        self.head_5kb = SingleResolutionHead(output_dim, 400, 400, hidden)
        
    def forward(self, pbulk_tracks):
        all_tracks = torch.cat([
            pbulk_tracks['fine'],
            pbulk_tracks['medium'],
            pbulk_tracks['coarse'],
        ], dim=1)
        
        all_tracks = all_tracks.unsqueeze(2)
        mixed_tracks = self.cross_track_mixer(all_tracks)
        
        feat_5kb = self.prepare_5kb(mixed_tracks).squeeze(2)
        feat_10kb = self.prepare_10kb(mixed_tracks).squeeze(2)
        feat_25kb = self.prepare_25kb(mixed_tracks).squeeze(2)
        
        predictions = {
            '25kb': self.head_25kb(feat_25kb),
            '10kb': self.head_10kb(feat_10kb),
            '5kb': self.head_5kb(feat_5kb)
        }
        
        return predictions

class Model_PbulkTopic_MultiRes(nn.Module):
    """Final 6-track HERTA model: Pbulk+CTCF branch + Topic branch."""
    def __init__(self):
        super(Model_PbulkTopic_MultiRes, self).__init__()
        
        self.branch_pbulk = Enhanced_Branch_Pbulk()
        self.branch_topic = branch_topic()
        
        self.multiresolution_decoder = MultiScaleFusionDecoder_6Track(
            single_track_dim=512,
            output_dim=1536,
            hidden=256,
            num_groups=16
        )

    def forward(self, x_pbulk, x_topic):
        pbulk_tracks = self.branch_pbulk(x_pbulk)
        topic_tracks = self.branch_topic(x_topic)
        
        predictions = self.multiresolution_decoder(
            pbulk_tracks,
            topic_tracks
        )
        
        return {'stage1': predictions}

class Model_PbulkOnly_MultiRes(nn.Module):
    """Stage-1 3-track Pbulk+CTCF pretraining model.

    """
    def __init__(self):
        super(Model_PbulkOnly_MultiRes, self).__init__()
        
        self.branch_pbulk = Enhanced_Branch_Pbulk()
        
        self.multiresolution_decoder = BulkOnlyMultiScaleFusionDecoder(
            single_track_dim=512,
            output_dim=1024,
            hidden=256,
            num_groups=16
        )

    def forward(self, x2):
        pbulk_tracks = self.branch_pbulk(x2)
        predictions = self.multiresolution_decoder(pbulk_tracks)
        return {'stage1': predictions}


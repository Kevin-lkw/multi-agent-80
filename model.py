import torch
from torch import nn
def get_model():
    return CNNLSTMModel()

class CNNModel(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self._tower = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1, bias = False),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1, bias = False),
            nn.ReLU(True),
            nn.Conv2d(256, 32, 3, 1, 1, bias = False),
            nn.ReLU(True),
            nn.Flatten()
        )
        self._logits = nn.Sequential(
            nn.Linear(32 * 4 * 14, 256),
            nn.ReLU(True),
            nn.Linear(256, 54)
        )
        self._value_branch = nn.Sequential(
            nn.Linear(32 * 4 * 14, 256),
            nn.ReLU(True),
            nn.Linear(256, 1)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input_dict):
        obs = input_dict["observation"].float()
        hidden = self._tower(obs)
        logits = self._logits(hidden)
        mask = input_dict["action_mask"].float()
        inf_mask = torch.clamp(torch.log(mask), -1e38, 1e38)
        masked_logits = logits + inf_mask
        value = self._value_branch(hidden)
        return masked_logits, value
    



class CNNLSTMModel(nn.Module):
    def __init__(self):
        super(CNNLSTMModel, self).__init__()
        # CNN to process ob_mat
        self._tower_ob_mat = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(256, 32, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Flatten()
        )
        
        # CNN to process each 3x4x14 matrix in seq_mat
        self._tower_seq_mat = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(128, 32, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Flatten()
        )
        
        # LSTM to process the sequence of embeddings from seq_mat
        self.lstm = nn.LSTM(32 * 4 * 14, 128, batch_first=True, bidirectional=False)
        
        # FFN to process the concatenated embeddings
        self._logits = nn.Sequential(
            nn.Linear((32 * 4 * 14) + 128, 256),
            nn.Tanh(),
            nn.Linear(256, 54)
        )
        
        self._value_branch = nn.Sequential(
            nn.Linear((32 * 4 * 14) + 128, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
        # Apply orthogonal initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, input_dict):
        ob_mat = input_dict["observation"].float()
        seq_mat = input_dict["seq_mat"]

        # Process ob_mat with CNN
        ob_embedding = self._tower_ob_mat(ob_mat)
        
        # Handle seq_mat processing
        if seq_mat.numel() != 0:
            # print(seq_mat.shape)
            seq_embeddings = []
            seq_mat = seq_mat.permute(1,0,2,3,4)
            for seq in seq_mat:
                seq_embedding = self._tower_seq_mat(seq.float())
                seq_embeddings.append(seq_embedding)
            seq_embeddings = torch.stack(seq_embeddings, dim=0)
            seq_embeddings = seq_embeddings.permute(1,0,2)
            # Process the sequence of embeddings with LSTM
            lstm_out, (h_n, c_n) = self.lstm(seq_embeddings)
            # print(h_n[-1].shape)
            lstm_embedding = h_n[-1]
        else:
            lstm_embedding = torch.zeros(ob_embedding.size(0), 128, device=ob_embedding.device)
        
        # Concatenate ob_embedding and lstm_embedding
        combined_embedding = torch.cat((ob_embedding, lstm_embedding), dim=1)
        
        # Get logits and value
        logits = self._logits(combined_embedding)
        value = self._value_branch(combined_embedding)
        
        # Apply action mask
        mask = input_dict["action_mask"].float()
        inf_mask = torch.clamp(torch.log(mask), -1e38, 1e38)
        masked_logits = logits + inf_mask
        
        return masked_logits, value



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=54):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(128, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._value_branch = nn.Sequential(
            nn.Linear(512 * block.expansion, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, input_dict):
        obs = input_dict["observation"].float()
        x = self.conv1(obs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        logits = self.fc(x)
        mask = input_dict["action_mask"].float()
        inf_mask = torch.clamp(torch.log(mask), -1e38, 1e38)
        masked_logits = logits + inf_mask
        
        value = self._value_branch(x)
        return masked_logits, value

def ResNet18(num_classes=54):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)



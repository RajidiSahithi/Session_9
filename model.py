import torch.nn.functional as F
import torch
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        drop=0.01
        # Convolutional Block 1
        self.conv11 = nn.Sequential (
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(drop)
            )
        self.conv12 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0,groups=4),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(drop)
            )
        self.conv13 = nn.Sequential(
            nn.Conv2d(16,16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(drop)
            )

        # Convolutional Block 2 
        self.conv21 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0,dilation=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            #nn.Dropout(drop),
           # )
        #self.conv22 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0,dilation=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            #nn.Dropout(drop),
           # )
        #self.conv23 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=9,dilation=3),
            nn.BatchNorm2d(16)#,
           # nn.ReLU(inplace=True),
            #nn.Dropout(drop)
                        )
        self.relu2 = nn.ReLU(inplace=True)
        #self.drop2 = nn.Dropout(drop)
         
        # Convolutional Block 3 (Depthwise)
        self.conv31 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(drop)
            )
        self.conv32 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(drop)
            )
        self.conv33 = nn.Sequential( #point wise Convolution
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(drop)
            )

        # Convolutional Block 4 (Dialation)
        self.conv41 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0,dilation=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            #nn.Dropout(drop),
            #)
        #self.conv42 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0,dilation=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            #nn.Dropout(drop),
            #)
        #self.conv43 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=9,dilation=3),
            nn.BatchNorm2d(64)#,
            #nn.ReLU(inplace=True),
            )
        self.relu1 = nn.ReLU(inplace=True)
        #self.drop1 = nn.Dropout(drop)

        self.gap = nn.AdaptiveAvgPool2d(1) 
        self.conv5 = nn.Sequential(  # Fully connected
            nn.Conv2d(64, 10, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, x):
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)

        s = x
        x = self.conv21(x)
        #x = self.conv22(x)
        #x = self.conv23(x)
        x+=s
        x = self.relu2(x)
        #x = self.drop2(x)
        
        x = self.conv31(x)
        x = self.conv32(x)
        x = self.conv33(x)
        residual = x
        x = self.conv41(x)
        #x = self.conv42(x)
        #x = self.conv43(x)
        x+=residual
        x = self.relu1(x)
        #x = self.drop1(x)

        x = self.gap(x)
        x = self.conv5(x)

        x = x.view(-1, 10)
        #return x
        y = F.log_softmax(x, dim=-1)
        return  y

def model_summary(model,input_size):
    model = Net().to(device)
    summary(model, input_size=(3, 32, 32))
    return model,input_size    
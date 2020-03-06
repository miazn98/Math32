from torch import nn
import math

def eulerNet():
        """
        Constructs a network for the base categorization model.

        Arguments
        --------
        netspec_opts: (dictionary), the network's architecture. It has the keys
                        'kernel_size', 'num_filters', 'stride', and 'layer_type'.
                        Each key holds a list containing the values for the
                        corresponding parameter for each layer.
        Returns
        ------
        net: (nn.Sequential), the base categorization model
        """
    # instantiate an instance of nn.Sequential
        net = nn.Sequential()
    #load the parameters from the

    # add layers as specified in netspec_opts to the network

      #   net = nn.Sequential(
      #   nn.Flatten(),
      #   nn.Linear(144, 128 ),
      #   nn.LeakyReLU(),
      #   nn.BatchNorm1d(128),
      #   nn.Linear(128, 512 ),
      #   nn.LeakyReLU(),
      #   nn.BatchNorm1d(512),
      #   nn.Dropout(p=0.2),
      #   nn.Linear(512, 256),
      #   nn.LeakyReLU(),
      #   nn.BatchNorm1d(256),
      #   nn.Dropout(p=0.5),
      #   nn.Linear(256, 8 ),
      #   nn.LeakyReLU(),
      #   nn.BatchNorm1d(8),
      #   nn.LeakyReLU(),
      #   nn.BatchNorm1d(1024),
      #   nn.Linear(1024, 8),
      #   )
      #   net = nn.Sequential(
      #   nn.Flatten(),
      #   nn.Linear(144, 64 ),
      #   nn.LeakyReLU(),
      #   nn.BatchNorm1d(64),
      #   nn.Linear(64, 128 ),
      #   nn.LeakyReLU(),
      #   nn.BatchNorm1d(128),
      #   nn.Dropout(p=0.2),
      #   nn.Linear(128, 256),
      #   nn.LeakyReLU(),
      #   nn.BatchNorm1d(256),
      #   nn.Dropout(p=0.5),
      #   nn.Linear(256, 8 ),
      #   nn.LeakyReLU(),
      #   nn.BatchNorm1d(8)
      #   )
      #   net = nn.Sequential(
      #   nn.Flatten(),
      #   nn.Linear(144, 64 ),
      #   nn.LeakyReLU(),
      #   nn.BatchNorm1d(64),
      #   nn.Linear(64, 128 ),
      #   nn.LeakyReLU(),
      #   nn.BatchNorm1d(128),
      #   nn.Dropout(p=0.2),
      #   nn.Linear(128, 256),
      #   nn.LeakyReLU(),
      #   nn.BatchNorm1d(256),
      #   nn.Dropout(p=0.5),
      #   nn.Linear(256, 8 ),
      #   nn.LeakyReLU(),
      #   nn.BatchNorm1d(8)
      #   )
        net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(144, 32 ),
            nn.LeakyReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 64 ),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(p=0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.5),
            nn.Linear(128, 8 ),
            nn.LeakyReLU(),
            nn.BatchNorm1d(8)
            )
        print("finished creating NN")

        return net

def cnn(netspec_opts):
   """
   Constructs a network for the improved categorization model.

   Arguments
   --------
   netspec_opts: (dictionary), the improved network's architecture.

   Returns
   -------
   A categorization model which can be trained by PyTorch
   """
   # instantiate an instance of nn.Sequential
   net = nn.Sequential()
   #load the parameters from the
   kernels = netspec_opts["kernel_size"]
   strides = netspec_opts["stride"]
   filters = netspec_opts["filters"]
   layer_types = netspec_opts["layer_type"]
   # add layers as specified in netspec_opts to the network
   in_channels = 0
   conv_ct = 0
   bn_ct = 0
   pool_ct = 0
   for layer in range(len(filters)):
      # padding = math.floor((kernels[layer]-1)/2) #k is assumed to be an odd integer. -- mod?
      padding = int((kernels[layer]-1)/2)

      if layer_types[layer] == "conv":
         conv_ct += 1
         name = layer_types[layer]+str(conv_ct)
         print(name)
         if layer==0:
               in_channels = 1
         padding = int((kernels[layer]-1)/2)
         net.add_module(name,
         nn.Conv2d(in_channels, filters[layer] , kernels[layer] , strides[layer],
               padding ) )
         #update in_channels to be the most recent filter
         in_channels = filters[layer]
      elif layer_types[layer]=="bn":
         name = layer_types[layer]+str(conv_ct)
         print(name)
         net.add_module(name, nn.BatchNorm2d(filters[layer]))
      elif layer_types[layer]=="relu":
         name = layer_types[layer]+str(conv_ct)
         print(name)
         net.add_module(name, nn.ReLU())
      elif layer_types[layer] == "pool":
         name = layer_types[layer]+str(conv_ct)
         print(name)
         padding = 0
         net.add_module(name, nn.MaxPool2d(kernels[layer] , strides[layer] , padding))
      elif layer_types[layer] == "pred":
         padding = 0
         name = "pred"
         print(name)
         net.add_module(name, #conv_{i}
         nn.Conv2d(in_channels, filters[layer] , kernels[layer] , strides[layer],
               padding ) )
   print("finished creating NN")

   return net

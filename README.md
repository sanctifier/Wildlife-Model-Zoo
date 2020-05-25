# Wildlife Model Zoo
Pytorch implementations of models designed for wildlife identification

Dependency requirements: math, collections, Pytorch > 1.1

The model can be employed with Pytorch > 0.4 by removing "align_corners=True" in downsample()

The model maintains the similar API as the builtin models in torchvision like AlexNet

To embed the model in your project or test script, you can place NASMP.py in your code file directory, and use the following statement to invoke the model:
  ```python
  import torchvision.datasets as dsets
  import torchvision.transforms as transforms
  from NASMP import NASMP
  GPU = "cuda:0"
  train_dataset = datasets.ImageFolder(...)
  train_loader = torch.utils.data.DataLoader(train_dataset, ...)
  model = NASMP(image_height, image_width, image_channel, image_class_number, GPU=GPU)
  model.cuda(device=GPU)
  model.train()
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  for epoch in range(num_epochs):
            correct = 0
            total = 0   
            for i, (images, labels) in enumerate(train_loader):                      
                optimizer.zero_grad()
                outputs = model(images.cuda(device=GPU))
                loss = criterion(outputs, labels.cuda(device=GPU))
                loss.backward()
                optimizer.step()
                if (i+1) % 100 == 0:
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted.cpu() == labels).sum()
                    print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f ACC: %.4f' 
                        %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.item(), (100 * correct / total)))
  ```

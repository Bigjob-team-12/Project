

model = ft_net(len(class_names), opt.droprate, opt.stride)

print(model)

ignored_params = list(map(id, model.classifier.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
optimizer_ft = optim.SGD([
    {'params': base_params, 'lr': 0.1*opt.lr},
    {'params': model.classifier.parameters(), 'lr': opt.lr}
    ], weight_decay=5e-4, momentum-0.9, nesterov=True)



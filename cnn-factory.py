

def load_pretrained_ResNet50(weights_path = None):
    weights_model = ResNet50(include_top=True, weights="imagenet", input_tensor=Input(shape=(224, 224, 3)), classes=1000)
    #weights_model.load_weights(weights_path)

    model = ResNet50(include_top=True, weights=None, input_tensor=Input(shape=(ROWS, COLS, 1)), classes=NB_CLASSES)
    for i in range(len(model.layers)):
        #print ("layer name: {}, input shape: {}, output shape:{}".format(model.layers[i].name, model.layers[i].input_shape, model.layers[i].output_shape))

        weights1 = weights_model.layers[i].get_weights()
        if len(weights1) > 0:
            if i == 1:
                # print('average the first conv layers over RGB channel for gray-scale use')
                print('use the first channel to initialize the first layer')
                average = [weights1[-1].mean(axis=2, keepdims=True)]
                # channel_1 = [weights1[-1][:,:,0:1,:]]
                model.layers[i].set_weights(average)
                model.layers[i].trainable = True
                continue
            if i == 312:
                print ('skip prediction layer')
                continue
            model.layers[i].set_weights(weights1)
    print('okay')
    weights_model = None
    for layer in model.layers[2:inception_start_train_layer]:
        layer.trainable = False
    for layer in model.layers[inception_start_train_layer:]:
        layer.trainable = True
    return model

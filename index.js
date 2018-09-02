//college_male, college_female, unknown
const NUM_CLASSES = 3;
let mobilenet;
let model;


async function init() {
  document.createElement('img');
  mobilenet = await loadMobilenet();
}

init();

//load mobile net from cdn
async function loadMobilenet() {
  const mobilenet = await tf.loadModel(
      'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

  // Return a model that outputs an internal activation.
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
};

let size = 0;

//triggered when user selects multiple images
function readFiles(evt) {
  let label = 0;
  let target = evt.target.id;
  //decide the label of the selected images based on the file chooser
  if(target == "college_female") {
    label = 0;
  } else if(target == "college_male") {
    label = 1;
  } else if(target == "sh_female") {
    label = 2;
  } else if(target == "sh_male") {
    label = 3;
  } else if(target == "unknown") {
    label = 2;
  }

  //read multiple images
  var files = evt.target.files;

    for(let i = 0; i < files.length; i++) {
      setTimeout(()=>{
        let img = new Image();

        let fr = new FileReader();
        fr.onload = function () {
          img.src = fr.result;
          //set image size to 224 x 224 to match input tensor to mobilenet
          img.width = 224;
          img.height = 224;
        }

        //TODO this must be synchronous
        img.onload = function() {
          tf.tidy(()=> {
            //convert to tensor
            const tf_img = cropImage(tf.fromPixels(img));
            const tf_img_batched = tf_img.expandDims(0);
            const tf_final_img_batched = tf_img_batched.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));

            addTrainingImage(mobilenet.predict(tf_final_img_batched), label);
          });
        };

        fr.readAsDataURL(files[i]);
      }, 500);
    }
}


//get the minimum dimension and crop the image based on that
function cropImage(img) {
    return tf.tidy(()=> {
      const size = Math.min(img.shape[0], img.shape[1]);
      const centerHeight = img.shape[0] / 2;
      const beginHeight = centerHeight - (size / 2);
      const centerWidth = img.shape[1] / 2;
      const beginWidth = centerWidth - (size / 2);
      return img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
    });
}

//push the training images with corresponding label to mobilenet
function addTrainingImage(trainingImage, label) {
  const y = tf.tidy(() => tf.oneHot(tf.tensor1d([label], 'int32'), NUM_CLASSES));

  if (this.xs == null) {
    this.xs = tf.keep(trainingImage);
    this.ys = tf.keep(y);
  } else {
    const oldX = this.xs;
    this.xs = tf.keep(oldX.concat(trainingImage, 0));

    const oldY = this.ys;
    this.ys = tf.keep(oldY.concat(y, 0));

    oldX.dispose();
    oldY.dispose();
    y.dispose();
  }
}

/**
 * Sets up and trains the classifier.
 */
async function train() {
  if (this.xs == null) {
    throw new Error('Add some examples before training!');
  }

  // Creates a 2-layer fully connected model. By creating a separate model,
  // rather than adding layers to the mobilenet model, we "freeze" the weights
  // of the mobilenet model, and only train weights from the new model.
  model = tf.sequential({
    layers: [
      // Flattens the input to a vector so we can use it in a dense layer. While
      // technically a layer, this only performs a reshape (and has no training
      // parameters).
      tf.layers.flatten({inputShape: [7, 7, 256]}),
      // Layer 1
      tf.layers.dense({
        units: 100,
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
        useBias: true
      }),
      // Layer 2. The number of units of the last layer should correspond
      // to the number of classes we want to predict.
      tf.layers.dense({
        units: NUM_CLASSES,
        kernelInitializer: 'varianceScaling',
        useBias: false,
        activation: 'softmax'
      })
    ]
  });

  // Creates the optimizers which drives training of the model.
  const optimizer = tf.train.adam(0.0001);
  // We use categoricalCrossentropy which is the loss function we use for
  // categorical classification which measures the error between our predicted
  // probability distribution over classes (probability that an input is of each
  // class), versus the label (100% probability in the true class)>
  model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});

  // We parameterize batch size as a fraction of the entire dataset because the
  // number of examples that are collected depends on how many examples the user
  // collects. This allows us to have a flexible batch size.
  const batchSize =
      Math.floor(this.xs.shape[0] * 0.4);
  if (!(batchSize > 0)) {
    throw new Error(
        `Batch size is 0 or NaN. Please choose a non-zero fraction.`);
  }

  // Train the model! Model.fit() will shuffle xs & ys so we don't have to.
  model.fit(this.xs, this.ys, {
    batchSize,
    epochs: 400,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        console.log('Loss: ' + logs.loss.toFixed(5));
      }
    }
  });
}

function readTestFiles(evt) {
  var files = evt.target.files;

  for(let i = 0; i < files.length; i++) {
    let fr = new FileReader();
    fr.onload = function () {
      img.src = fr.result;
    }

    img.onload = function() {
      tf.tidy(()=> {
        //convert to tensor
        const tf_img = cropImage(tf.fromPixels(img));
        const tf_img_batched = tf_img.expandDims(0);
        const tf_final_img_batched = tf_img_batched.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));

        const predictedClass = tf.tidy(() => {
          const activation = mobilenet.predict(tf_final_img_batched);
          const predictions = model.predict(activation);
          predictions.as1D().print();
          return predictions.as1D().argMax();
        });
      });
    };

    fr.readAsDataURL(files[i]);
  }


}


document.getElementById('college_female').addEventListener('change', readFiles, false);
document.getElementById('college_male').addEventListener('change', readFiles, false);
// document.getElementById('college_male').addEventListener('sh_female', readFiles, false);
// document.getElementById('college_male').addEventListener('sh_male', readFiles, false);
document.getElementById('unknown').addEventListener('change', readFiles, false);

document.getElementById('fileinputTest').addEventListener('change', readTestFiles, false);

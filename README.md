# ASSIGNMENT_3
# Character-level Neural Machine Transliteration (Seq2Seq) with Attention

This repository contains an implementation of a character-level neural machine transliteration model with attention mechanism using PyTorch. The model is designed to transliterate English words into Hindi. 

## Overview

The transliteration model consists of an encoder-decoder architecture with and without attention mechanism. The encoder processes the input English sequence, while the decoder generates the corresponding Hindi transliteration character by character. The attention mechanism allows the decoder to focus on different parts of the input sequence during the generation process.
link to wandb report ()

## Dataset
the [Aksharantar](https://drive.google.com/file/d/1uRKU4as2NlS9i8sdLRS1e326vQRdhvfw/view?pli=1) dataset released by AI4Bharat was used in the project.This dataset contains pairs of the following form: 
xx x,yy y
ajanabee,अजनबी
i.e., a word in the native script and its corresponding transliteration in the Latin script (how we type while chatting with our friends on WhatsApp etc). Given many such (xi,yi)i=1n(x_i, y_i)_{i=1}^n
(xi
​
,yi
​
)i=1
n
​

 pairs your goal is to train a model y=f^(x)y = \hat{f}(x)
y=f
^
​
(x)
 which takes as input a romanized string (ghar) and produces the corresponding word in Devanagari (घर).Refer this [blog](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) for pytorch for understanding 


## building a RNN model

1. Preprocess the data:

   ```bash
   python preprocess.py --input data/english.txt --output data/hindi.txt
   ```

2. Train the model:

   ```bash
   python train.py --source data/english.txt --target data/hindi.txt --epochs 20 --batch_size 64
   ```

3. Evaluate the model:

   ```bash
   python evaluate.py --model models/model.pth --source data/english_test.txt --target data/hindi_test.txt
   ```

4. Transliterate new input:

   ```bash
   python transliterate.py --model models/model.pth --input "Hello, world!"
   ```

## Model Configuration

The model architecture, including the number of layers, hidden units, and attention mechanism, can be configured by modifying the `config.py` file.

## Results

The model achieves an accuracy of X% on the validation set and demonstrates good performance in transliterating English words to Hindi.

## Contributing

Contributions to this project are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Acknowledgments

This implementation is based on the work by [reference paper or author].

## References

[Include any relevant papers or articles here]

Please refer to the [Wiki](https://github.com/your-username/neural-machine-transliteration/wiki) for more detailed information on the project and its components.

## Authors

- [Your Name](https://github.com/your-username)

Feel free to contact the author with any questions or inquiries.

## Release History

- 0.1.0
  - Initial release

## TODO

- [ ] Add visualization of attention weights
- [ ] Improve handling of out-of-vocabulary words
- [ ] Add support for other languages

## FAQ

[Include any frequently asked questions here]

For a more detailed explanation of the project and its components, please refer to the [project documentation](https://github.com/your-username/neural-machine-transliteration/blob/main/docs/README.md).

**Note:** It is important to ensure that you have sufficient computational resources for training the model, as it may require significant memory and processing power depending on the size of the dataset and the chosen model configuration.

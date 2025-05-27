# llama fine tuning project for comment generation

A fine-tuned version of Vikhrmodels/Vikhr-Llama3.1-8B-Instruct-R-21-09-24 for comment generation.


## Table of Contents
- Installation
- Usage
- Fine-Tuning Details
- Troubleshooting
- Contributing
- License
- Acknowledgments
- Contact


## Installation
0. I am assumming that you are using linux
1. Clone the repository
2. Options to pip install
    - If you want to gather the data yourself then pip install `requirements.txt` in the root (Also setup your postgreSQL db and telegram pyrogram session yourself)
    - If you want to just fine tune the model then you need to pip install `docker/requirements.txt`
3. Place your training (for comment generation only) json data in `data/processed_data/`
    - The data should look like in the file `data/data_types/processed_data/example_data.json`
4. Download the model that you want in `scripts/download_tokenizer_and_model.py`
5. Change `scripts/fine_tune_scripts/config.py` file for your preferences
6. Change Dockerfile in `docker/` and make docker image
    - You should change data location and model location
    - Also check `docker/docker_commands.md` and `docker/docker_setup.md` (should be useful)


## Usage
1. **Access Options**  
   - Download from Hugging Face:  
     - [`mccarryster/com-gen-llama3.1-8B`](https://huggingface.co/mccarryster/com-gen-llama3.1-8B)  
     - [`mccarryster/com-gen-llama3.2-1B`](https://huggingface.co/mccarryster/com-gen-llama3.2-1B)
2. In the `scripts/model_inference.py` file you can check:
    - How to load the model from hugging face
    - Prompt
    - Tokenize prompt
    - Make inference


## Fine-Tuning Details
1. Base model
    - Vikhrmodels/Vikhr-Llama-3.2-1B-Instruct: Trained on Russian lanague and small sized
    - Vikhrmodels/Vikhr-Llama3.1-8B-Instruct-R-21-09-24: Trained on Russian lanague and nice performance
2. Dataset
    - Source: Telegram channel
    - Size: 152425
    - Structure: Input post and output comment pairs (Russian language)
    - Preprocessing: Removed too short and too long comments
    - Evaluation:
        - Final training loss: 2.9202
        - Final perplexity: 18.5451
        - Human Evaluation: The generated comments were reviewed manually, and the results were satisfactory.


## Troubleshooting
1. Generation Quality Issues
    - Tune generation parameters
2. Performance and Speed
    - Using too long prompt will obviosly slows generation
3. Memory Limitations
    - Consider that llama3.1 8B loaded in bf16 uses ~15GB of VRAM


## Contributing
If you find something wrong or wierd in my code
1. Bug Reports & Feature Requests: Open an issue describing the problem or feature you want to see. Include as much detail as possible to help understand the request
2. Code Contributions:
    - Fork the repository and create a new branch for your changes
    - Write clear, concise commit messages
3. Pull Requests:
    - Submit your PR against the main branch
    - Describe what your changes do and why they are needed
4. Documentation:
    - Improvements to documentation are very welcome


## License
This project is licensed under the MIT License.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


## Acknowledgments
Thanks to the developers of the original Llama model for making their pre-trained weights publicly available, which served as the foundation for this fine-tuned comment generation model
Thanks to the open-source community for providing invaluable tools and libraries such as PyTorch, Hugging Face Transformers


## Contact
1. Mail: mccarryster.lev@yahoo.com
2. Twitter (X): @mccarryster
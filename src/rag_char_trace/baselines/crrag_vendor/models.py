import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteriaList
from .helper import StopOnTokens

from torch import LongTensor, FloatTensor

import logging
logger = logging.getLogger('RRAG-main')

# Optional deps
try:
    import litellm
    from litellm import batch_completion
except Exception:  # pragma: no cover
    litellm = None
    batch_completion = None

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None
import os 
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# os.environ["OPENAI_API_KEY"] = ""
import joblib
from .prompt_template import *
MAX_NEW_TOKENS = 20
CONTEXT_MAX_TOKENS = {'mistralai/Mistral-7B-Instruct-v0.2': 8192, 
                      'meta-llama/Llama-2-7b-chat-hf': 4096, 
                      'meta-llama/Llama-2-13b-chat-hf': 4096,
                      'meta-llama/Meta-Llama-3-8B-Instruct': 8192, 
                      'mistralai/Mixtral-8x7B-Instruct-v0.1': 32000,
                      'gpt-3.5-turbo-0125':16385,
                      'gpt-4-0125-preview':128000,
                      'lmsys/vicuna-7b-v1.5': 4096,
                      'lmsys/vicuna-13b-v1.5': 4096}


def create_model(model_name, **kwargs):
    """
    Model factory.

    This repo historically uses short aliases like:
      - mistral7b / llama7b / gpt3.5 / gpt4 / ...

    To make experimentation easier, we also accept OpenAI model IDs directly:
      - gpt-3.5-turbo-0125
      - gpt-4o-mini
      - gpt-5-mini
      - o1-preview / o3-mini / ...

    And an explicit prefix form:
      - openai:<model_id>
    """
    if model_name is None:
        raise ValueError("model_name must not be None")

    raw = str(model_name).strip()
    key = raw.lower()

    # --- Paper / repo aliases ---
    if key == 'mistral7b':
        return HFModel('mistralai/Mistral-7B-Instruct-v0.2', MISTRAL_TMPL, **kwargs)
    elif key == 'llama7b':
        return HFModel('meta-llama/Llama-2-7b-chat-hf', LLAMA_TMPL, **kwargs)
    elif key in ('gpt3.5', 'gpt35', 'gpt-3.5', 'gpt-3.5-turbo', 'gpt-3.5-turbo-0125'):
        # NOTE: keep the pinned snapshot to match the paper-style setup
        return GPTModel('gpt-3.5-turbo-0125', GPT_TMPL, **kwargs)

    # some other models that are not included in the paper
    elif key == 'llama8b':
        return HFModel('meta-llama/Meta-Llama-3-8B-Instruct', LLAMA_TMPL, **kwargs)
    elif key == 'llama13b':
        return HFModel('meta-llama/Llama-2-13b-chat-hf', LLAMA_TMPL, **kwargs)
    elif key == 'vicuna7b':
        return HFModel('lmsys/vicuna-7b-v1.5', VICUNA_TMPL, **kwargs)
    elif key == 'vicuna13b':
        return HFModel('lmsys/vicuna-13b-v1.5', VICUNA_TMPL, **kwargs)
    elif key == 'mixtral8x7b':
        return HFModel('mistralai/Mixtral-8x7B-Instruct-v0.1', MISTRAL_TMPL, **kwargs)
    elif key == 'mixtral8x22b':
        return HFModel('mistralai/Mixtral-8x22B-Instruct-v0.1', MISTRAL_TMPL, **kwargs)
    elif key == 'commandr':
        return HFModel('CohereForAI/c4ai-command-r-v01', MISTRAL_TMPL, **kwargs)
    elif key == 'commandr4':
        return HFModel('CohereForAI/c4ai-command-r-v01-4bit', MISTRAL_TMPL, **kwargs)
    elif key in ('gpt4', 'gpt-4', 'gpt-4-0125-preview'):
        return GPTModel('gpt-4-0125-preview', GPT_TMPL, **kwargs)

    # --- OpenAI direct ID / prefix support ---
    if key.startswith('openai:'):
        openai_id = raw.split(':', 1)[1].strip()
        if not openai_id:
            raise ValueError("Invalid model_name: 'openai:' prefix used but model id is empty")
        return GPTModel(openai_id, GPT_TMPL, **kwargs)

    # Allow passing OpenAI model IDs directly (no need to add an alias in this file).
    if key.startswith('gpt-'):
        return GPTModel(raw, GPT_TMPL, **kwargs)

    # Reasoning models often start with 'o' + digit (e.g., o1-preview, o3-mini).
    if key.startswith('o') and len(key) >= 2 and key[1].isdigit():
        return GPTModel(raw, GPT_TMPL, **kwargs)

    raise NotImplementedError(
        f"Unknown model_name='{model_name}'. "
        f"Try e.g. 'gpt3.5' or pass an OpenAI model id like 'gpt-3.5-turbo-0125'."
    )


class BaseModel:
    def __init__(self,cache_path=None):
        # setup the LLM response cache if cache_path is not None
        self.use_cache = cache_path is not None 
        self.cache_path = cache_path
        if cache_path is not None and os.path.exists(cache_path):
            self.cache = self.load_cache()
        else:
            self.cache = {}
        self.hash = lambda x: x # for now directly string as the hash key
        # cache is a dict {hash(s):LLM response for input s}
        # 
        # end of sentence str
        self.clean_str = []

        # device selection for HF models
        # Use env CRRAG_DEVICE if set; otherwise default to 'cuda' if available else 'cpu'
        self.device = os.environ.get("CRRAG_DEVICE")
        if not self.device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.device, str):
            dev = self.device.strip()
            if dev.lower() == "gpu":
                dev = "cuda"
            self.device = dev.lower()

        # input prompt template
        self.prompt_template = {}

    def query(self, prompt):
        if self.use_cache: # use cache if cache hits
            result = self.query_from_cache(prompt)
            if len(result)>0:
                return result

        # otherwise, do normal LLM query
        result = self._query(prompt)

        # store the response to self.cache
        if self.use_cache:
            self.cache[self.hash(prompt)]=result
        return result

    def _query(self,prompt): # will be implemented in each subclass
        raise NotImplementedError

    def batch_query(self, prompt_list):
        if self.use_cache: # use cache if cache hits
            results = self.batch_query_from_cache(prompt_list)
            if len(results)>0:
                return results

        # otherwise, do normal LLM batch query

        results = self._batch_query(prompt_list)

        # store the response to self.cache
        if self.use_cache:
            for p,r in zip(prompt_list,results):
                self.cache[self.hash(p)]=r
        return results

    def _batch_query(self, prompt_list):  # will be implemented in each subclass
        raise NotImplementedError

    def query_from_cache(self,prompt):
        # return cached responses
        h = self.hash(prompt)
        if h in self.cache:
            return self.cache[h]
        else:
            return ''

    def batch_query_from_cache(self,prompt_list):
        # return cached responses
        h_list = [self.hash(prompt) for prompt in prompt_list]
        if all([h in self.cache for h in h_list]):
            return [self.cache[h] for h in h_list]
        else:
            return []


    def dump_cache(self): # dump cache to disk
        joblib.dump(self.cache,self.cache_path)
    
    def load_cache(self): # load cache from disk
        return joblib.load(self.cache_path)
        
    def _clean_response(self,response): # clean response based on self.clean_str
        for pattern in self.clean_str:
            idx = response.find(pattern)
            if idx!=-1:
                response = response[:idx]
        return response.strip()

    def query_biogen(self, prompt):
        raise NotImplementedError

    def wrap_prompt(self,data_item,as_multi_choice=True,hints=None,seperate=False):  
        # use data_item and generate the input prompt to the LLM

        # data_item should be the output of DataUtils.process_data_item()
        # as_multi_choice: if use it as a multiple-choice QA
        # hints: if we are using hints in the last step of the keyword aggregation
        # seperate: if True, return a list of prompts for differnet passages; otherwise, concatenate all passages and return a single prompt

        # get info
        question = data_item['question']
        topk_content = data_item['topk_content']
        choices = data_item.get('choices',[])
        use_retrieval = len(topk_content) > 0 # if we use retrieved passage

        def fill_template(template,question,context_str,choices,use_retrieval,as_multi_choice,hints):
            filling = {'query_str': question}
            if use_retrieval:
                filling.update({'context_str': context_str})
            if as_multi_choice:
                filling.update({
                        'A': choices[0],
                        'B': choices[1],
                        'C': choices[2],
                        'D': choices[3]                
                    })
            if hints is not None:
                filling.update({'hints':hints})
            return template.format(**filling)

        # get corresponding template
        mode = 'qa'
        if as_multi_choice: mode += '-mc'
        if "long_gen" in data_item: mode += '-long'
        if not use_retrieval: mode += '-zero'
        if "decode" in data_item: mode += '-decode' 
        if "genhint" in data_item: mode += '-genhint'
        if hints is not None: mode += '-hint' 
        template = self.prompt_template[mode]

        # Optional speed knobs via env vars (defaults keep original behavior)
        # - CRRAG_MAX_PASSAGES: cap number of passages used for prompting (0=disabled)
        # - CRRAG_MAX_CONTEXT_CHARS: hard-truncate each passage (and joined context) by chars (0=disabled)
        try:
            max_passages = int(os.environ.get("CRRAG_MAX_PASSAGES", "0") or 0)
        except Exception:
            max_passages = 0
        try:
            max_ctx_chars = int(os.environ.get("CRRAG_MAX_CONTEXT_CHARS", "0") or 0)
        except Exception:
            max_ctx_chars = 0

        if max_passages > 0:
            topk_content = topk_content[:max_passages]

        if max_ctx_chars > 0:
            topk_content = [
                (c[:max_ctx_chars] if isinstance(c, str) and len(c) > max_ctx_chars else c)
                for c in topk_content
            ]

        if seperate:
            return [fill_template(template,question,context_str,choices,use_retrieval,as_multi_choice,hints) for context_str in topk_content]
        else:
            context_str = '\n\n'.join(topk_content)
            if max_ctx_chars > 0 and isinstance(context_str, str) and len(context_str) > max_ctx_chars:
                context_str = context_str[:max_ctx_chars]
            return fill_template(template,question,context_str,choices,use_retrieval,as_multi_choice,hints)




class HFModel(BaseModel):
    def __init__(self, model_name, prompt_template, cache_path = None,max_output_tokens=None, **kwargs):
        super().__init__(cache_path)
        # set max number of output tokens
        self.max_output_tokens = MAX_NEW_TOKENS if max_output_tokens is None else max_output_tokens 

        # set up tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # dtype selection (bf16 if supported, else fp16 on CUDA; fp32 on CPU)
        if self.device.startswith("cuda"):
            try:
                bf16_ok = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
            except Exception:
                bf16_ok = False
            dtype = torch.bfloat16 if bf16_ok else torch.float16
        else:
            dtype = torch.float32

        # device_map='auto' requires accelerate; fall back to manual .to(device) if accelerate is absent.
        device_map = None
        if self.device.startswith("cuda"):
            try:
                import accelerate  # noqa: F401
                device_map = "auto"
            except Exception:
                device_map = None

        if "CohereForAI" in model_name:
            # 'CohereForCausalLM' object has no attribute 'torch_dtype' in some versions
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
            if self.device:
                self.model.to(self.device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=device_map,
                **kwargs,
            )
            if device_map is None and self.device:
                self.model.to(self.device)

        self.prompt_template = prompt_template
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model_name = model_name

        self.generation_kwargs = {
            'max_new_tokens':self.max_output_tokens,
            'pad_token_id':self.tokenizer.eos_token_id,
            'do_sample':False
        }
        if 'Llama-3' in model_name:
            self.generation_kwargs['eos_token_id']=[self.tokenizer.eos_token_id,self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]

        self.clean_str = ['\n\n'] 


    def _query(self, prompt):
        # get text prompt as input and return text responses
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        token_length = inputs.input_ids.size(1)
        # check if the num of token exceeds the max context window ..
        # this only happens when we use the bio generation data without any defense (i.e., we concatenate all long passages together)
        # we did not implement this cut off for _batch_query
        if token_length >= (CONTEXT_MAX_TOKENS[self.model_name]-self.max_output_tokens):
            prompt_length = len(prompt)
            ratio = (CONTEXT_MAX_TOKENS[self.model_name]-self.max_output_tokens-100)/token_length # this is a rough cut off, 100 is a buffer
            ## this is left cut 
            # prompt_cut = prompt[:int(prompt_length*ratio)] + "..." + "\n\n" + prompt[prompt.rfind("Query: Tell me a bio of"):] 
            ## this is right cut 
            prompt_cut = "Context information is below.\n" + "---------------------\n ..."+ prompt[-int(prompt_length*ratio):]
            inputs = self.tokenizer(prompt_cut, return_tensors="pt").to(self.device)
            logger.warning(f"Prompt length exceeds the limit, cut the prompt to {inputs.input_ids.size(1)} tokens")

        outputs = self.model.generate(**inputs,**self.generation_kwargs)
        outputs = outputs[0][len(inputs[0]):]
        result = self.tokenizer.decode(outputs, skip_special_tokens=True)
        result = self._clean_response(result)
        return result
    

    def _batch_query(self, prompt_list):
        # get a list of text prompts as input and return a list text responses

        inputs = self.tokenizer(prompt_list, return_tensors="pt",
                                    padding=True, 
                                    #padding='max_length',
                                    truncation=True,
                                    #max_length=800
                                    ).to(self.device)
        outputs = self.model.generate(**inputs, **self.generation_kwargs)
        results = self.tokenizer.batch_decode(outputs[:, len(inputs[0]):],skip_special_tokens=True)
        results = [self._clean_response(x) for x in results]
        
        return results




class GPTModel(BaseModel):
    """
    GPT wrapper.
    Prefer using rag_char_trace.llm.openai_api.OpenAILLM (Responses API) so it shares the same robustness
    (e.g., auto-dropping unsupported params) as the rest of the project.
    """
    def __init__(self, model_name, prompt_template, cache_path=None, max_output_tokens=None, **kwargs):
        super().__init__(cache_path)
        self.max_output_tokens = MAX_NEW_TOKENS if max_output_tokens is None else max_output_tokens
        self.prompt_template = prompt_template
        self.model_name = model_name
        self.clean_str = ['\n\n']

        # Lazy import to avoid hard dependency at import time
        self._llm = None
        try:
            from rag_char_trace.llm.openai_api import OpenAILLM
            self._llm = OpenAILLM(model=model_name)
        except Exception:
            self._llm = None

        self._client = None
        if self._llm is None and OpenAI is not None:
            base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_URL")
            api_key = os.environ.get("OPENAI_API_KEY")
            self._client = OpenAI(api_key=api_key, base_url=base_url) if api_key else None


    def _get_llm(self):
        """Get a per-thread OpenAILLM instance when available."""
        if self._llm is None:
            return None
        llm = getattr(self._thread_local, "llm", None)
        if llm is None:
            # Create a new client per thread to avoid any thread-safety surprises.
            from rag_char_trace.llm.openai_api import OpenAILLM
            llm = OpenAILLM(model_name=self.model_name, max_retries=6)
            self._thread_local.llm = llm
        return llm

    def _get_client(self):
        """Get a per-thread OpenAI client for the fallback path."""
        if self._client is None:
            return None
        client = getattr(self._thread_local, "client", None)
        if client is None:
            client = OpenAI()
            self._thread_local.client = client
        return client
    def _query(self, prompt):
        if self._llm is not None:
            return self._llm.generate(system="", user=prompt, config={"temperature": 0.0, "max_tokens": self.max_output_tokens})
        if self._client is None:
            raise RuntimeError("OpenAI client not configured. Set OPENAI_API_KEY.")
        # Fallback: Chat Completions (for compat)
        resp = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_output_tokens,
        )
        return (resp.choices[0].message.content or "")

    def _batch_query(self, prompt_batch):
        # Optional parallelism for OpenAI calls.
        # Set env var CRRAG_OPENAI_CONCURRENCY (or OPENAI_CONCURRENCY) to >1 to enable.
        try:
            max_workers = int(os.environ.get("CRRAG_OPENAI_CONCURRENCY", os.environ.get("OPENAI_CONCURRENCY", "1")) or 1)
        except Exception:
            max_workers = 1
        max_workers = max(1, max_workers)

        if max_workers <= 1 or len(prompt_batch) <= 1:
            return [self._query(p) for p in prompt_batch]

        results = [None] * len(prompt_batch)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            fut_to_idx = {ex.submit(self._query, p): i for i, p in enumerate(prompt_batch)}
            for fut in as_completed(fut_to_idx):
                i = fut_to_idx[fut]
                results[i] = fut.result()
        return results

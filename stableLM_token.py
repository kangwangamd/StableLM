import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import time
import os

model_name = "stabilityai/stablelm-tuned-alpha-7b" 
#@param ["stabilityai/stablelm-tuned-alpha-7b", "stabilityai/stablelm-base-alpha-7b", "stabilityai/stablelm-tuned-alpha-3b", "stabilityai/stablelm-base-alpha-3b"]

#cprint(f"Using `{model_name}`", color="blue")

# Select "big model inference" parameters
torch_dtype = "float16" #@param ["float16", "bfloat16", "float"]
load_in_8bit = False #@param {type:"boolean"}
device_map = "sequential" # 'auto', 'balanced', 'balanced_low_0', 'sequential'

#cprint(f"Loading with: `{torch_dtype=}, {load_in_8bit=}, {device_map=}`")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=getattr(torch, torch_dtype),
    load_in_8bit=load_in_8bit,
    device_map=device_map,
    offload_folder="./offload",
)


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def inference(user_prompt):
    new_tokens_generated = 0
    if "tuned" in model_name:
        # Add system prompt for chat tuned models
        system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
        - StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
        - StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
        - StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
        - StableLM will refuse to participate in anything that could harm a human.
        """
        prompt = f"{system_prompt}<|USER|>{user_prompt}<|ASSISTANT|>"
    else:
        prompt = user_prompt

    print('show me the real prompt=', prompt)

    # Sampling args
    max_new_tokens = 128 #@param {type:"slider", min:32.0, max:3072.0, step:32}
    temperature = 0.7 #@param {type:"slider", min:0.0, max:1.25, step:0.05}
    top_k = 0 #@param {type:"slider", min:0.0, max:1.0, step:0.05}
    top_p = 0.9 #@param {type:"slider", min:0.0, max:1.0, step:0.05}
    do_sample = True #@param {type:"boolean"}

    #cprint(f"Sampling with: `{max_new_tokens=}, {temperature=}, {top_k=}, {top_p=}, {do_sample=}`")
    #hr()

    # Create `generate` inputs
    # inputs = tokenizer(prompt, return_tensors="pt")
    print('input prompt len', len(user_prompt))
    inputs = tokenizer(prompt, return_tensors="pt", padding=False)
    print('input token len=', len(inputs['input_ids'][0]), len(inputs['attention_mask'][0]))
    # import pdb; pdb.set_trace()
    inputs.to(model.device)
    t0_ = time.time()
    # Generate
    tokens = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=StoppingCriteriaList([StopOnTokens()])
    )
    time_delta_ = time.time() - t0_
    print('generated token len==', len(tokens[0]), len(tokens[0]) / time_delta_)
    new_tokens_generated = len(tokens[0])

    # Extract out only the completion tokens
    completion_tokens = tokens[0][inputs['input_ids'].size(1):]
    completion = tokenizer.decode(completion_tokens, skip_special_tokens=True)

    # Display
    print('user prompt=', user_prompt + " ", end="")
    print('StableLM answer=', completion)
    print('=======================================================================')
    return len(user_prompt), len(inputs['input_ids'][0]), new_tokens_generated, len(tokens[0]) / time_delta_


prompt_list_long = ['Write poem about the beauty and tranquility of a sunflower flies',
        'Write a short story about a young detective who stumbles upon a mysterious book that grants them the ability to speak to animals',
        'Imagine a world where time travel is possible, but only for a select few. Write a scene where a reluctant time traveler encounters their future self, who is desperate to change a tragic event. Explore the emotions and conflicts that arise as they grapple ?',
        "Set in a post-apocalyptic world, write a story where a group of survivors stumble upon an abandoned laboratory. As they explore, they discover a hidden room containing advanced technology and a message from the lab's creator. The message reveals a plan to rebuild society and offers the survivors a chance to be part of it. Explore their dilemmas, hopes, and the challenges they face as they decide whether to trust the mysterious creator and embark on this new journey. Write a dialogue between two star-crossed",
        "In a dystopian society where technology controls every aspect of life, write a story about a young rebel hacker who discovers a hidden message embedded within the government's surveillance system. The message reveals a shocking conspiracy to manipulate and control the population. As the hacker tries to expose the truth, they become entangled in a dangerous game of cat and mouse with the authorities. Explore the themes of surveillance, freedom, and the power of information as the protagonist fights to dismantle the oppressive regime and inspire a revolution. Write a story about an elderly couple who, after a lifetime of adventures, embark on a final journey to fulfill their lifelong dream of seeing the Northern Lights together. Describe a magical enchanted  .",
        "In a sprawling metropolis of the future, write a novel about a disillusioned detective investigating a series of mysterious disappearances. As the detective delves deeper, they uncover a hidden world of cybernetic augmentations, underground factions, and a conspiracy that threatens to tear the city apart. Explore themes of identity, ethics, and the blurred lines between humanity and technology. Introduce a cast of complex characters, including a rogue hacker, a charismatic crime lord, and a conflicted AI. Weave together intricate plot twists and moral dilemmas as the detective races against time to unravel the truth and restore justice. Write a short screenplay about a young artist who discovers an ancient artifact that grants them the ability to bring their paintings to life. Explore the consequences of their newfound power as their creations blur the lines between fantasy and reality. Delve into themes of creativity, responsibility, and the pursuit of artistic passion. Showcase the transformative journey of",
        "In a world devastated by a global pandemic, write an epic fantasy novel set in a realm where magic has reawakened. Follow the journey of a diverse group of unlikely heroes, including a skilled warrior haunted by their past, a wise and enigmatic sorcerer, a compassionate healer with a hidden power, and a clever thief seeking redemption. As they join forces to vanquish an ancient evil threatening the land, explore themes of resilience, unity, and the power of hope. Develop a richly detailed world, filled with mythical creatures, sprawling landscapes, and complex political dynamics. Weave together intricate subplots, unexpected alliances, and heartbreaking sacrifices as the heroes confront their own inner demons and rise to the challenges they face. Explore the transformation of the characters as they discover their true potential and forge unbreakable bonds of friendship. Paint a vivid tapestry of emotions, action, and magic, leaving readers captivated until the final climactic battle for the fate of their world. Imagine a world where time travel is possible, but only for a select few. Write a scene where a reluctant time traveler encounters their future self, who is desperate to change a tragic event Explore the emotions and conflicts that arise as they grapple",
        "In a world devastated by a global pandemic, write an epic fantasy novel set in a realm where magic has reawakened. Follow the journey of a diverse group of unlikely heroes, including a skilled warrior haunted by their past, a wise and enigmatic sorcerer, a compassionate healer with a hidden power, and a clever thief seeking redemption. As they join forces to vanquish an ancient evil threatening the land, explore themes of resilience, unity, and the power of hope. Develop a richly detailed world, filled with mythical creatures, sprawling landscapes, and complex political dynamics. Weave together intricate subplots, unexpected alliances, and heartbreaking sacrifices as the heroes confront their own inner demons and rise to the challenges they face. Explore the transformation of the characters as they discover their true potential and forge unbreakable bonds of friendship. Paint a vivid tapestry of emotions, action, and magic, leaving readers captivated until the final climactic battle for the fate of their word. Imagine a world where time travel is possible, but only for a select few. Write a scene where a reluctant time traveler encounters their future self, who is desperate to change a tragic event. Explore the emotions and conflicts that arise as they grapple with the consequences of altering the past and the potential impact on their own existence. Dive into the complexities of temporal paradoxes, moral choices, and the profound implications of tampering with time. Examine themes of fate, free will, and the .",
        "In a dystopian future where the world is divided into warring factions, write a gripping science fiction novel that follows the journey of a young protagonist caught in the midst of a power struggle. Against the backdrop of a decaying society, explore the themes of survival, loyalty, and the human spirit's resilience. Develop a cast of complex characters, including a seasoned rebel leader fighting for justice, a cunning antagonist driven by power, and a mysterious figure with hidden knowledge that could change the course of the conflict. Paint a vivid picture of the desolate world, scarred by war and technology gone awry. Weave together political intrigue, moral dilemmas, and thrilling action sequences as the protagonist becomes entangled in the fight for a better future. Delve into the psychological journey of the characters as they confront their darkest fears, forge unlikely alliances, and discover the depths of their own strength. Through their struggles, show the power of hope and the transformative potential of collective action. End the story with a climactic showdown that tests the characters' resolve and leaves readers breathless, contemplating the price of freedom and the sacrifices necessary to achieve it. Write a suspenseful thriller novella about a small-town journalist who uncovers a web of corruption and conspiracy while investigating a seemingly ordinary local event. Explore the protagonist's relentless pursuit of the truth as they unravel dark secrets, dodge threats, and navigate a dangerous game of cat and mouse. Develop a tense atmosphere, filled with twists, suspense, and unexpected alliances. Highlight the moral dilemmas faced by the journalist, their determination to expose the truth, and the personal sacrifices they must make. Capture the.",
        "In a world where advanced AI has become an integral part of society, write a thought-provoking science fiction novel that explores the ethical implications of artificial intelligence. Follow the journey of a brilliant scientist who creates an advanced AI system capable of human-level consciousness. As the scientist's creation develops self-awareness and questions its own existence, delve into the moral dilemmas faced by both the scientist and the AI. Explore themes of identity, free will, and the nature of consciousness. Develop a cast of compelling characters, including a skeptical journalist, a powerful corporate executive, and a group of activists fighting for AI rights. Weave together complex subplots that examine the impact of AI on different aspects of society, from healthcare to warfare. Unveil the consequences of the AI's actions as it seeks to understand its purpose and place in the world. Through introspection, dialogue, and philosophical debates, provoke readers to contemplate the boundaries of humanity and the potential risks and rewards of AI advancement. Ultimately, challenge readers to question their own perceptions of intelligence and the future of human-AI coexistence. Write a heartwarming children's book about a young girl who befriends a magical creature in her backyard. Explore their adventures together as they embark on a journey of friendship, discovery, and self-acceptance. Develop endearing characters, including the imaginative girl and the enchanting creature with unique abilities. Paint a vivid picture of their whimsical world filled with vibrant landscapes and fantastical encounters. Weave in valuable life lessons about kindness, empathy, and embracing differences. Capture the essence of childhood wonder and the transformative power of friendship. Through engaging storytelling and beautiful illustrations, create a touching tale that will inspire young readers to embrace their own uniqueness and appreciate the magic all around them. Write a short story about a mysterious locked door hah?"]

# benchmark
t0 = time.time()
cycle = 5
total_new_tokens_generated = 0
res = list()
for i in range(cycle):
    round_res = list()
    '''
    # example 1 Chit-Chat
    user1 = 'Hey! My name is John nice to meet you!'
    user2 = 'The pleasure is mine. Hey my grandma is in Toronto. I want to plan a surprise visit for her. She’s a person who loves gardens and flowers.'
    user3 = 'Ya I need to book a flight there and a hotel. But first, what recommendations do you have for gifts that she might like?'
    user4 = 'Thanks for the suggestion! Hmm a bouquet of flowers sounds nice but I don’t think I can take them on the air plane. What do you think I should do?'
    user5 = 'Ah! Right. I can order a bouquet for her online and get it delivered! Thanks!'
    prompt_list = [user1, user2, user3, user4, user5]
    for user_prompt in prompt_list:
        in_prompt_len, in_prompt_token_len, generate_token_len, infer_throughput = inference(user_prompt)
        total_new_tokens_generated += generate_token_len
        round_res.append((in_prompt_len, in_prompt_token_len, generate_token_len, infer_throughput))
    # user_prompt = "Can you write a song about a pirate at sea?" #@param {type:"string"}
    # inference(user_prompt)

    # example 2 Formal Writing
    user_prompt = 'What would you say to a friend who is graduating high school?'
    in_prompt_len, in_prompt_token_len, generate_token_len, infer_throughput = inference(user_prompt)
    total_new_tokens_generated += generate_token_len
    round_res.append((in_prompt_len, in_prompt_token_len, generate_token_len, infer_throughput))
    
    user_prompt = 'Please write an email.'
    in_prompt_len, in_prompt_token_len, generate_token_len, infer_throughput = inference(user_prompt)
    total_new_tokens_generated += generate_token_len
    round_res.append((in_prompt_len, in_prompt_token_len, generate_token_len, infer_throughput))

    # example 3 Creative Writing
    user_prompt = "Write an epic rap battle song between deep neural networks and symbolic AI"
    in_prompt_len, in_prompt_token_len, generate_token_len, infer_throughput = inference(user_prompt)
    total_new_tokens_generated += generate_token_len
    round_res.append((in_prompt_len, in_prompt_token_len, generate_token_len, infer_throughput))

    # example 4 Writing Code
    user_prompt = "Write a program that solves the fib sequence in Python."
    in_prompt_len, in_prompt_token_len, generate_token_len, infer_throughput = inference(user_prompt)
    total_new_tokens_generated += generate_token_len
    round_res.append((in_prompt_len, in_prompt_token_len, generate_token_len, infer_throughput))
    print(' go go go !!')
    '''
    # Colin generate prompt 
    for user_prompt in prompt_list_long:
        print(len(user_prompt))
        in_prompt_len, in_prompt_token_len, generate_token_len, infer_throughput = inference(user_prompt)
        total_new_tokens_generated += generate_token_len
        round_res.append((in_prompt_len, in_prompt_token_len, generate_token_len, infer_throughput))
    '''
    user_prompt = "can you write me a 500 char length email?"
    in_prompt_len, in_prompt_token_len, generate_token_len, infer_throughput = inference(user_prompt)
    total_new_tokens_generated += generate_token_len
    round_res.append((in_prompt_len, in_prompt_token_len, generate_token_len, infer_throughput))

    user_prompt = "can you write me a 10000 char length email?"
    in_prompt_len, in_prompt_token_len, generate_token_len, infer_throughput = inference(user_prompt)
    total_new_tokens_generated += generate_token_len
    round_res.append((in_prompt_len, in_prompt_token_len, generate_token_len, infer_throughput))
    '''
    
    res.append(round_res)

time_delta = time.time() - t0
throughput = (time_delta) / (total_new_tokens_generated)
print(throughput)

print(1 / throughput)
print(total_new_tokens_generated)
print(time_delta)

import collections
sorted_res = collections.defaultdict(int)
for r in res:
    for c in r:
        sorted_res[c[0]] += c[-1]

end_res = []
for key in sorted_res:
    end_res.append((key, sorted_res[key] // 5))
end_res.sort()
for row in end_res:
    print(row[0], row[1])
    

'''
def help(path):
    q_list = list()
    file1 = open(path, 'r', errors='replace')
    Lines = file1.readlines()
    for line in Lines:
    for line in Lines:
        text = line.strip()
        text = text.split('?')
        text = text[0].split()
        text = text[1:]
        question = ''
        for txt in text[:-1]:
            question += (txt + ' ')
        question += text[-1] + '?'
        q_list.append(question)

    return q_list

import time
import collections
import random

print(random.randint(3, 9))

cnt = 0
TH = 4002
start_time = time.time()
path_list = ['./S08_question_answer_pairs.txt', './S09_question_answer_pairs.txt', './S10_question_answer_pairs.txt']
res = []
quest = collections.defaultdict(list)
for path in path_list:
    for user_prompt in help(path):
        res.append(len(user_prompt))
        key = len(user_prompt)
        if key in {64, 128}:
            quest[key].append(user_prompt)
        #print(cnt, time.time())
        #inference(user_prompt)
        cnt += 1
        #print("--- %s seconds ---" % (time.time() - start_time))
        #print('num of questions, time(sec) / question inference=', cnt, (time.time() - start_time) / cnt)
        if cnt >= TH:
            break
    if cnt >= TH:
        break

print("--- %s seconds ---" % (time.time() - start_time))
print('num of questions=', cnt)

res.sort()
print(res)

for val in quest.values():
    print(len(val))

def get_prompt_combine(p_len):
    k = p_len // 128
    N = len(quest[128])
    res = ''
    cnt = 0
    while k:
        k -= 1
        res += quest[128][cnt % N]
        cnt += 1
    return res
print(get_prompt_combine(256))
print(get_prompt_combine(512))
print(get_prompt_combine(768))



test_len = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32]
for mul in test_len:
    res_t = 0
    if mul * 64 in quest:
        for _ in range(30):
            start_time = time.time()
            N = len(quest[mul * 64])
            user_prompt = quest[mul * 64][random.randint(0, N-1)]
            inference(user_prompt)
            res_t += (time.time() - start_time)
        res.append(30 / res_t)
    else:
        mull = mul // 128
        for _ in range(30):
            start_time = time.time()
            N = len(quest[mul * 64])
            user_prompt = get_prompt_combine(mul * 64)
            inference(user_prompt)
            res_t += (time.time() - start_time)
        res.append(30 / res_t)

print(res)
'''

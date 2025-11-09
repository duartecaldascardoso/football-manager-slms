# The Underrated Role of Small Language Models in Agentic AI

---

I recently stumbled upon a thought evoking research paper from NVIDIA that challenged how I view AI agents. The premise was simple: we are probably using models that are way too big for what most AI agents actually need to do.

So I decided to test this hypothesis myself. I built a multi-agent system to manage a football game - complete with commentary, score tracking, and player management - using nothing but small language models running locally. The results were aligned with the idea from the article.

## The Question That Started Everything

It is known that LLMs are groundbreaking and that they excel in powerful use cases.
However, most use cases do not need such power or scale. Thinking about it - when you build an agent to create/update/delete an element - you are asking it to do the same handful of tasks over and over again. It does not need to know about most information in the countless billion parameters it was trained on.

The NVIDIA paper laid out three compelling reasons why Small Language Models (SLMs) might actually be better for these use cases:

- They are powerful enough for specialized tasks
- Agentic systems naturally break down into focused sub-tasks anyway
- The economics of running huge models for simple tasks just do not make sense

## Building the Experiment

For fun, I designed a football game management system with three AI agents working together:

- The **Supervisor Agent** acts as the coordinator, figuring out which specialized agent should handle each request.
- The **Commenting Agent** handles everything related to game commentary, player information, and coach details.
- The **Game Manager Agent** takes care of the operational stuff - creating games, updating scores, and managing player expulsions.

Of course, this is pretty basic game interaction, but it serves the experiment due to the number of tools per agent, simulating a real world use case.

For the model, I chose Qwen 3:8B - a small language model with 8 billion parameters, that can use tools. Everything was ran locally using Ollama, meaning no API calls, no cloud dependencies, just my GPU burning my lap :/ 

## The Architecture

The system uses the supervisor architecture, mentioned in the LangChain docs: https://docs.langchain.com/oss/javascript/langchain/supervisor#2-create-specialized-sub-agents. The supervisor receives all incoming requests and routes them to the right agent. Each agent has its own set of tools - the commenting agent can look up active players, check who's been expelled, and generate creative commentary ideas. The game manager can create new games, update scores, and kick players out.

I used LangChain and LangGraph to orchestrate everything, with LangGraph's MessagesState handling the conversation history. This way, the agents could maintain context across multi-turn conversations without me having to manually track everything.

## What Actually Happened

### Good agent behaviour

The model handled tool selection very well. The tools were well described and simple, following most tool description best practices. The supervisor correctly routed commentary requests to the Commenting Agent and game management tasks to the Game Manager Agent. Individual agents picked the right tools from their toolkits without confusion.

This validated what I wanted to test: these kinds of models are capable of dealing with such scenarios, and can deal with most of the use cases, even if only handling a sub set of tasks from the wider range that can be handled by LLMs.

### The Verbosity Surprise

Something I did not see coming was that the model was concise. This brought me joy, because I have been feeling like other LLM models have been feeling context windows with useless information and verbosity.

Direct, focused responses that got straight to the point is the best way to describe it. 

This has practical benefits too:
- Responses come back faster (fewer tokens to generate)
- Users do not have to navigate AI bloat to get to the answer
- Lower costs from reduced output tokens (in this case it is actually 0, because I am running it locally :) ).

## When This Approach Works (And When It Doesn't)

Being the advocate for the devil, SLMs knowingly are not the answer to everything. My experiment worked well in the focused situation,  I gave the agents specialized, well-defined tasks, with up to 5 tools per agent.

However, there are places where LLMs most likely have to be used (the article also states that):
- Complex multistep reasoning chains that require deep logical deduction
- Questions that need extensive world knowledge
- Open-ended creative tasks where you are not sure what you are looking for
- Highly ambiguous queries that need a lot of context to interpret

General-purpose conversational assistants probably still benefit from LLMs. If you need the latest information about current events, you will want a bigger model with more recent training data (or a good tool to browse the web). And for applications where response quality is critical, and you have low request volumes, the cost of LLMs might be justified.

The optimal solution for many real-world applications is probably a hybrid approach, use SLMs when possible, but keep an LLM available for complex edge cases or as a supervisor handling sophisticated routing decisions.

## What I Learned

A few things became clear through this experiment:

- **Small models are more capable than we give them credit for.** Qwen 3:8B handled tool calling and parameter generation with the kind of accuracy I expected from much larger models. This challenges the assumption that you need frontier models for production-grade agentic systems.

- **The economics are compelling enough to matter.** A 94% cost reduction isn't a marginal improvement - it's a completely different cost structure. For any organization running agents at scale, this is worth serious consideration.

- **Less verbose isn't worse.** The industry has somehow equated "more words" with "better quality," but in agentic applications, conciseness is often preferable. SLMs get to the point faster, and that's actually a feature.

- **On-premises deployment opens doors.** Running AI entirely locally means organizations in regulated industries can adopt agentic AI without the compliance headaches of cloud APIs. That's not a small thing.

## Final Thoughts

This experiment started as a way to validate a research paper's claims, but it turned into something more - a realization that we might be over-engineering our AI solutions out of habit rather than necessity.

Small language models running locally on modest hardware can power sophisticated multi-agent systems with high accuracy, low latency, and a fraction of the cost. They're more concise, easier to secure, and simpler to deploy. For many - maybe most - agentic applications, they're not just sufficient. They're better.

The AI industry moves fast, and we're often chasing the biggest, most capable models. But sometimes the right tool for the job is the smaller one you can actually run on your laptop.

---

**Tools and Technologies:**
- Qwen 3:8B - The SLM powering all agents
- LangChain - Agent orchestration framework
- LangGraph - State management
- Ollama - Local model deployment
- Python - Implementation language

**References:**
- "Small Language Models are the Future of Agentic AI" (2025). NVIDIA LPR Team. arXiv:2506.02153
- LangChain Documentation: https://docs.langchain.com/

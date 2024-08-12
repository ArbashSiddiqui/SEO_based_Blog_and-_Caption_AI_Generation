import os
import random
import nltk
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from tavily import TavilyClient
import tiktoken
from langchain.schema import Document as LangChainDocument
from nltk.corpus import wordnet
from rake_nltk import Rake
from prompts import blog_prompt, summary_prompt, caption_prompt,title_prompt, vocab, topicwords

# Load environment variables from the .env file for API keys and other configurations
load_dotenv()

class ArticleProcessor:
    def __init__(self):
        # Get API keys from environment variables
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")

        # Ensure both API keys are provided
        if not tavily_api_key or not openai_api_key:
            raise ValueError("API keys must be provided in the .env file")

        # Set API keys in environment for other API client usage and initialize the language model
        os.environ["TAVILY_API_KEY"] = tavily_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)

        # Define prompt templates for generating summaries, blogs, captions, and titles
        self.prompt_templates = {
            "summary": PromptTemplate(
                input_variables=["text"],
                template=summary_prompt
            ),
            "blog": PromptTemplate(
                input_variables=["summary", "text"],
                template=blog_prompt
            ),
            "caption": PromptTemplate(
                input_variables=["summary"],
                template=caption_prompt
            ),
            "title": PromptTemplate(
                input_variables=["query"],
                template=title_prompt
            )
        }

        # Initialize the client for searching articles using Tavily API
        self.client = TavilyClient(api_key=tavily_api_key)

    def search_articles(self, query, max_results=5):
        Urls = [(result['url']) for result in response['results']]
        response = self.client.search(query, max_results=max_results, include_raw_content=True)
        articles = [(result['url'], result['raw_content']) for result in response['results']]
        content = "\n".join(article[1] for article in articles)
        return articles, content , Urls

    def create_chain(self, prompt_template):
        return LLMChain(llm=self.llm, prompt=prompt_template)

    def process_text(self, text, prompt_type):
        chain = self.create_chain(self.prompt_templates[prompt_type])
        return chain.run(text)

    # def save_text_to_word(self, text, filename):
    #     doc = Document()
    #     doc.add_heading('Blog Post', level=1)
    #     paragraphs = text.split('\n\n')
    #     humanized_paragraphs = []
    #     vocabulary = vocab
    #     topic_words = topicwords
    #     for paragraph in paragraphs:
    #         sentences = nltk.sent_tokenize(paragraph)
    #         humanized_sentences = [self.paraphrase_sentence(sentence, excluded_words, topic_words, vocabulary) for sentence in sentences]
    #         humanized_paragraphs.append(' '.join(humanized_sentences))

    #     return '\n\n'.join(humanized_paragraphs)

    def paraphrase_sentence(self, sentence, excluded_words=[], topic_words=[], vocabulary=[]):
        # Paraphrase sentences by replacing words with synonyms that are not in the excluded list
        words = nltk.word_tokenize(sentence)
        paraphrased_words = []

        for word in words:
            synonyms = wordnet.synsets(word)
            suitable_synonyms = [syn.lemmas()[0].name() for syn in synonyms if syn.lemmas()[0].name() not in excluded_words and syn.lemmas()[0].name() in vocabulary]
            if suitable_synonyms:
                synonym = random.choice(suitable_synonyms)
                paraphrased_words.append(synonym)
            else:
                paraphrased_words.append(word)

        return ' '.join(paraphrased_words)

    def process_query(self, query):
        _, content = self.search_articles(query)
        summary = self.process_text(content, "summary")
        blog_post = self.process_text({"summary": summary, "text": content}, "blog")
        caption = self.process_text(summary, "caption")
        
        r = Rake()
        r.extract_keywords_from_text(blog_post)
        keywords = r.get_ranked_phrases()  # You might use keywords further if needed

        vocabulary = vocab
        topic_words = topicwords
        excluded_words = ["arsenic", "angstrom"]
        humanized_blog_post = self.humanize_text(blog_post, excluded_words, topic_words, vocabulary)

        filename_after = f"{query}_after.docx"
        self.save_text_to_word(humanized_blog_post, filename_after)

        print(f"Paraphrased blog post saved as {filename_after}")

if __name__ == "__main__":
    # Main execution block to take user input and process the query
    processor = ArticleProcessor()
    query = input("Enter the topic name: ")
    results = processor.process_query(query)
    # # print(f"Summary: {results[0]}")
    # print(f"Blog Post: {results[1]}")
    # # print(f"Caption: {results[2]}")
    # # print(f"Title: {results[3]}")

summary_prompt='''Find the key points and updated info in the following text:

The summary should be precise and paraphrased.
Incorporate conversational language, slang, and idiomatic expressions that a human might use naturally
Infuse the text with personality, emotions, and subjective opinions.
Use expressive language to convey emotions and feelings, making the text more engaging and relatable.
The system should generate an SEO-optimized summary.
Text:\n\n{text}\n\nSummary'''


blog_prompt=''' You are an expert SEO-optimized blog writer. Based on the summary and text.
Generate a detailed blog post based on the provided summary and text.
The blog should be SEO-optimized with bold headings and subheadings related to the topic. The content should be accurate and informative.
Use a mix of short, medium, and long sentences. Vary sentence structures to break the monotony and make the text feel more human-writteanecdotes, stories, and specific examples. Infuse the text with subjective opinions and expressive language to convey emotions and feelings.
Infuse the text with personality, emotions, and subjective opinions.
Use expressive language to convey emotions and feelings, making the text more engaging and relatable.
Ensure headings and subheadings are SEO-optimized. Use relevant keywords naturally within the content. Focus on readability and user engagement.Generate the blog using the provided content but it should be paraphrased.
Keep in mind that the generated content should not sound like AI-generated.
The generated content should sound like human-written text and use keywords with in the content of Blog.
Introduce minor grammar mistakes to mimic human writing. Check for redundancy and rephrase sentences to maintain a natural flow.Summary and Text:\n\n{summary}\n\n{text}\n\nBlog Post:
Response must be in a SEO-Optimized Blog post format.

'''

caption_prompt='''Create a caption based on the following summary:\n"
                         "1. The caption should be of 3 to 4 lines and must be SEO-optimized.\n"
                         "2. Use many relevant hashtags that should also be SEO-optimized and trending.\n"
                         "Summary:\n\n{summary}\n\nCaption:'''
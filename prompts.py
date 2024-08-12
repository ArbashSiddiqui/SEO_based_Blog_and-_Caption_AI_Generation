summary_prompt = '''
Summarize the key points and recent updates in the text below. The summary should:
- Be concise and paraphrased.
- Use conversational language, including slang and idiomatic expressions.
- Include personality and subjective opinions.
- Express emotions and make the text engaging and relatable.
- Be SEO-optimized.
Text:\n\n{text}\n\nSummary:
'''



blog_prompt = '''
Create an expert, SEO-optimized blog post based on the summary and text provided. The blog should:
- Be detailed and informative, using a variety of sentence structures.
- Include anecdotes, personal stories, and specific examples.
- Express subjective opinions and emotions.
- Feature bold headings and subheadings that are SEO-optimized.
- Introduce minor grammatical imperfections to mimic human writing.
- Ensure the content does not sound AI-generated but instead appears human-written.
- Incorporate relevant keywords naturally.
- write a blog like you are talking and explanaing to a person.
Summary and Text:\n\n{summary}\n\n{text}\n\nBlog Post:
'''


caption_prompt = '''
Create a caption based on the summary provided. The caption should:
- Be 3 to 4 lines long and SEO-optimized.
- Include trending and relevant hashtags.
- Sound conversational and relatable.
Summary:\n\n{summary}\n\nCaption:
'''
title_prompt='''Generate a new title using {query}.
-Title must be SEO optimized.
-Title should not exceed one line (max 7 to 8 words).
-Title must be catchy.
'''

vocab = [
    "insight", "perspective", "explore", "discover", "community", "growth", "transformation", 
    "experience", "journey", "innovation", "creativity", "collaboration", "passion", "dedication", 
    "sustainability", "impact", "global", "inclusive", "diverse", "vision", "opportunity", "engagement", 
    "support", "empowerment", "connection", "leadership", "values", "future", "challenge", "solutions", 
    "change", "potential", "development", "strategy", "inspiration", "well-being", "success", "motivation", 
    "knowledge", "wisdom", "culture", "tradition", "heritage", "responsibility", "trust", "integrity", 
    "respect", "empathy", "compassion", "innovation", "progress", "achievement", "celebration", "joy", 
    "hope", "resilience", "strength", "aspiration", "balance", "harmony", "adventure", "curiosity", 
    "discovery", "wonder", "enrichment", "understanding", "clarity", "focus", "mindfulness", "serenity", 
    "peace", "wellness", "gratitude", "appreciation", "kindness", "unity", "togetherness", "support", 
    "encouragement", "persistence", "tenacity", "ambition", "drive", "enthusiasm", "pride", "humility", 
    "confidence", "self-awareness", "adaptability", "flexibility", "resourcefulness", "initiative", 
    "entrepreneurship", "innovation", "creativity", "design", "expression", "artistry", "craftsmanship", 
    "excellence", "mastery", "skill", "competence", "expertise", "professionalism", "knowledge", "insight", 
    "understanding", "wisdom", "learning", "education", "development", "training", "mentorship", "guidance", 
    "coaching", "support", "advocacy", "inclusion", "diversity", "equality", "fairness", "justice", 
    "respect", "dignity", "humanity", "compassion", "empathy", "care", "concern", "consideration", "responsibility", 
    "accountability", "ethics", "integrity", "honesty", "transparency", "trust", "faith", "belief", "confidence", 
    "hope", "optimism", "positivity", "enthusiasm", "energy", "vitality", "health", "well-being", "fitness", 
    "nutrition", "diet", "lifestyle", "balance", "harmony", "peace", "serenity", "tranquility", "calm", 
    "relaxation", "recreation", "leisure", "fun", "enjoyment", "pleasure", "happiness", "joy", "delight", 
    "satisfaction", "contentment", "gratitude", "appreciation", "thankfulness", "recognition", "acknowledgment", 
    "respect", "admiration", "praise", "commendation", "reward", "celebration", "success", "achievement", 
    "accomplishment", "victory", "triumph", "conquest", "glory", "honor", "fame", "renown", "reputation", 
    "prestige", "status", "standing", "position", "rank", "title", "role", "duty", "responsibility", 
    "obligation", "commitment", "promise", "pledge", "vow", "oath", "declaration", "assertion", "statement", 
    "claim", "opinion", "view", "belief", "conviction", "principle", "value", "ethic", "standard", "norm", 
    "criterion", "benchmark", "yardstick", "measure", "indicator", "sign", "symbol", "emblem", "token", 
    "representation", "illustration", "example", "instance", "case", "model", "prototype", "pattern", "template", 
    "framework", "structure", "system", "method", "approach", "strategy", "plan", "program", "project", 
    "initiative", "campaign", "effort", "endeavor", "enterprise", "venture", "activity", "task", "job", 
    "work", "operation", "function", "role", "duty", "responsibility", "service", "support", "assistance", 
    "aid", "help", "benefit", "advantage", "gain", "profit", "reward", "return", "outcome", "result", 
    "consequence", "effect", "impact", "influence", "force", "power", "control", "authority", "leadership", 
    "management", "direction", "guidance", "supervision", "oversight", "regulation", "governance", 
    "administration", "organization", "coordination", "collaboration", "partnership", "teamwork", 
    "cooperation", "unity", "solidarity", "harmony", "synchronization", "integration", "connection", 
    "relationship", "association", "affiliation", "bond", "link", "tie", "union", "alliance", "coalition", 
    "federation", "network", "system", "structure", "framework", "foundation", "base", "platform", 
    "support", "basis", "ground", "cornerstone", "keystone", "pillar", "backbone", "mainstay", "linchpin"
]

topicwords = [
    "innovation", "development", "learning", "skills", "opportunities", "benefits", "market", "tools", 
    "platforms", "technology", "user", "needs", "requirements", "design", "testing", "reliability", 
    "challenges", "creativity", "expertise", "productivity", "effectiveness", "projects", "platform", 
    "real-world scenarios", "problem-solving", "automation", "job displacement", "ethical dilemmas", 
    "future", "possibilities", "synergy", "trend", "landscape", "embrace", "transformation", "boost", 
    "flexibility", "skills", "stand out", "professional network", "certification", "opportunities", 
    "endeavor", "journey", "innovation", "game-changer", "competitive edge"
]

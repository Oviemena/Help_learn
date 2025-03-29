# Technology categories with their related terms and resources
TECH_CATEGORIES = {
    'keywords': {
        'frontend': ['react', 'vue', 'angular', 'javascript', 'html', 'css', 'ui', 'frontend'],
        'backend': ['python', 'java', 'nodejs', 'api', 'server', 'database', 'backend'],
        'devops': ['docker', 'kubernetes', 'aws', 'ci/cd', 'deployment', 'cloud'],
        'system_design': ['architecture', 'scalability', 'distributed', 'microservices'],
        'data_science': ['machine learning', 'data analysis', 'statistics', 'ai']
    },
    'resources': {
        'documentation': [
            'docs.{tech}.dev', 
            'docs.{tech}.org', 
            '{tech}.dev', 
            '{tech}.org'
        ],
        'learning': [
            'learn.{tech}.dev',
            'tutorial.{tech}.com',
            'freecodecamp.org',
            'dev.to'
        ],
          'video': [
            'youtube.com',
            'coursera.org',
            'udemy.com'
        ],
        'community': [
            'github.com',
            'stackoverflow.com',
            'medium.com'
        ]
    }
}

# Generic learning path templates
LEARNING_PATHS = {
    'beginner': {
        'templates': [
            "Learn {topic} fundamentals and core concepts",
            "Practice basic {topic} with hands-on exercises",
            "Build simple {topic} projects"
        ],
        'keywords': ['beginner tutorial', 'basics', 'fundamentals', 'getting started']
    },
    'intermediate': {
        'templates': [
            "Master {topic} patterns and best practices",
            "Build real-world {topic} applications",
            "Implement advanced {topic} features"
        ],
        'keywords': ['intermediate', 'practical', 'real-world examples']
    },
    'advanced': {
        'templates': [
            "Design complex {topic} architectures",
            "Optimize {topic} performance",
            "Contribute to {topic} ecosystem"
        ],
        'keywords': ['advanced', 'optimization', 'architecture', 'expert']
    }
}

def get_resources_for_topic(topic, level):
    """Dynamically generate resource domains based on topic and level"""
    topic_lower = topic.lower()
    
    # Find matching category
    category = next(
        (cat for cat, keywords in TECH_CATEGORIES['keywords'].items() 
         if any(kw in topic_lower for kw in keywords)),
        'general'
    )
    
    # Level-specific resource domains
    level_domains = {
        'beginner': {
            'priorities': ['learning', 'documentation', 'community'],
            'additional': ['freecodecamp.org', 'w3schools.com', 'codecademy.com']
        },
        'intermediate': {
            'priorities': ['documentation', 'community', 'learning'],
            'additional': ['medium.com/topic/programming', 'dev.to']
        },
        'advanced': {
            'priorities': ['community', 'documentation', 'learning'],
            'additional': ['github.com/topics', 'papers.arxiv.org']
        }
    }
    
    # Get level-specific configuration
    level_config = level_domains.get(level, level_domains['beginner'])
    
    # Generate prioritized resource domains
    domains = []
    
    # Add category-specific resources first
    if category != 'general':
        for resource_type in level_config['priorities']:
            patterns = TECH_CATEGORIES['resources'][resource_type]
            domains.extend([
                pattern.format(tech=topic_lower.split()[0])
                for pattern in patterns
            ])
    
    # Add level-specific additional resources
    domains.extend(level_config['additional'])
    
    # Add general community resources
    domains.extend(TECH_CATEGORIES['resources']['community'])
    
    return list(set(domains))  # Remove duplicates
"""
Response Template Generator
Provides appropriate response templates for each comment category
"""

class ResponseTemplates:
    """
    Generate contextual response templates for different comment categories
    """
    
    @staticmethod
    def get_templates():
        """
        Get all response templates organized by category
        
        Returns:
            dict: Dictionary of category -> list of templates
        """
        templates = {
            'Praise': [
                "Thank you so much for your kind words! 🙏 Your support means the world to us!",
                "We're thrilled you enjoyed it! Thanks for the amazing feedback! ❤️",
                "Your appreciation keeps us motivated! Thank you for being awesome! 🌟",
                "So glad you loved it! More exciting content coming your way! 🚀",
                "Thank you! Comments like yours make all the hard work worthwhile! 💪"
            ],
            
            'Support': [
                "Your support means everything! Thank you for believing in us! 🙏",
                "We're so grateful for encouragement like yours! Thank you! ❤️",
                "Thank you for being part of our journey! Your words inspire us! 🌟",
                "Your belief in us keeps us going! Much appreciated! 💪",
                "Thanks for the positive energy! We won't let you down! 🚀"
            ],
            
            'Constructive Criticism': [
                "Thank you for the honest feedback! We'll definitely work on improving that. 🙏",
                "We really appreciate your constructive input! This helps us grow. 📈",
                "Thanks for taking the time to share your thoughts! We'll consider this for future work. 💡",
                "Great point! We're always looking to improve and your feedback is valuable. ✨",
                "Thank you for the thoughtful critique! We'll use this to enhance our next project. 🎯"
            ],
            
            'Hate': [
                "We appreciate all feedback. If you have specific concerns, we're here to listen. 🙏",
                "Sorry you feel that way. We're always working to improve. Thanks for watching. ✨",
                "We respect your opinion. Hope you'll give our future content a chance. 🌟",
                "Thank you for your feedback. We're committed to doing better. 💪",
                "[Moderate/Ignore] - Document for review if pattern continues."
            ],
            
            'Threat': [
                "[ESCALATE TO MODERATION] - Screenshot and report to platform immediately.",
                "[DO NOT ENGAGE] - Block user and document threat for legal team.",
                "[FLAG FOR REVIEW] - This comment requires immediate attention from safety team.",
                "We take threats seriously. This has been reported to the appropriate authorities.",
                "[LEGAL ACTION PROTOCOL] - Preserve evidence and notify legal department."
            ],
            
            'Emotional': [
                "We're so touched that this resonated with you! Thank you for sharing! ❤️",
                "Your heartfelt comment means so much to us! Sending you positive vibes! 🌟",
                "Thank you for being vulnerable and sharing your feelings! We appreciate you! 🙏",
                "We're honored this connected with you emotionally! Thank you! 💫",
                "Your emotional response reminds us why we create! Thank you for watching! ✨"
            ],
            
            'Spam': [
                "[DELETE/IGNORE] - Mark as spam and remove comment.",
                "[BLOCK USER] - Report to platform as spam/promotional content.",
                "[NO RESPONSE NEEDED] - Filter automatically if possible.",
                "[HIDE COMMENT] - Use platform tools to reduce visibility.",
                "[REPORT] - Flag for platform spam detection."
            ],
            
            'Question': [
                "Great question! [Provide specific answer here] 🙏",
                "Thanks for asking! We'll address this in detail. Stay tuned! 💡",
                "Love this question! [Answer or] Check our FAQ/bio for more info! ✨",
                "Excellent question! We'll create content specifically about this! 🎯",
                "Thanks for your curiosity! [Direct answer or resource] 📚"
            ],
            
            'Suggestion': [
                "Love this idea! We'll definitely consider it for future content! 💡",
                "Great suggestion! We're taking note of this! 📝",
                "Thanks for the awesome idea! This could be really interesting! ✨",
                "We appreciate the suggestion! Always open to creative input! 🎯",
                "Interesting concept! We'll explore how we can incorporate this! 🚀"
            ]
        }
        
        return templates
    
    @staticmethod
    def get_response(category, template_index=0):
        """
        Get a specific response template for a category
        
        Args:
            category (str): Comment category
            template_index (int): Index of template to use
            
        Returns:
            str: Response template
        """
        templates = ResponseTemplates.get_templates()
        
        if category not in templates:
            return "Thank you for your comment! We appreciate your engagement. 🙏"
        
        category_templates = templates[category]
        index = template_index % len(category_templates)
        
        return category_templates[index]
    
    @staticmethod
    def get_all_responses(category):
        """
        Get all response templates for a category
        
        Args:
            category (str): Comment category
            
        Returns:
            list: List of all templates for the category
        """
        templates = ResponseTemplates.get_templates()
        return templates.get(category, ["Thank you for your comment! 🙏"])
    
    @staticmethod
    def get_action_guide():
        """
        Get action guidelines for each category
        
        Returns:
            dict: Category -> action guide
        """
        guide = {
            'Praise': {
                'priority': 'High',
                'action': 'Respond warmly and quickly',
                'engagement': 'Like, heart, or pin comment',
                'notes': 'These are your brand advocates - nurture them!'
            },
            'Support': {
                'priority': 'High',
                'action': 'Acknowledge and thank personally',
                'engagement': 'Heart the comment, respond with gratitude',
                'notes': 'Build community by recognizing supporters'
            },
            'Constructive Criticism': {
                'priority': 'Very High',
                'action': 'Respond thoughtfully and professionally',
                'engagement': 'Acknowledge feedback, show willingness to improve',
                'notes': 'Most valuable for growth - take seriously!'
            },
            'Hate': {
                'priority': 'Medium',
                'action': 'Respond once politely, then ignore/hide',
                'engagement': 'Do not argue or over-engage',
                'notes': 'Document pattern for potential blocking'
            },
            'Threat': {
                'priority': 'Critical',
                'action': 'Do NOT respond - escalate immediately',
                'engagement': 'Report, block, document for legal team',
                'notes': 'Safety first - involve platform and authorities'
            },
            'Emotional': {
                'priority': 'High',
                'action': 'Respond with empathy and care',
                'engagement': 'Heart comment, validate their feelings',
                'notes': 'Creates deep audience connection'
            },
            'Spam': {
                'priority': 'Low',
                'action': 'Delete/hide without response',
                'engagement': 'Report as spam, block repeat offenders',
                'notes': 'Use automated filters when possible'
            },
            'Question': {
                'priority': 'High',
                'action': 'Answer clearly and helpfully',
                'engagement': 'Pin if commonly asked, create FAQ content',
                'notes': 'Opportunity to provide value and create content'
            },
            'Suggestion': {
                'priority': 'High',
                'action': 'Thank them and consider seriously',
                'engagement': 'Like comment, credit if implemented',
                'notes': 'Free market research - valuable insights!'
            }
        }
        
        return guide
    
    @staticmethod
    def get_priority_order():
        """
        Get recommended order for responding to comments
        
        Returns:
            list: Categories in priority order
        """
        return [
            'Threat',              # Critical - immediate action
            'Constructive Criticism',  # Very High - important feedback
            'Question',            # High - needs answers
            'Suggestion',          # High - valuable input
            'Praise',              # High - acknowledge support
            'Support',             # High - nurture community
            'Emotional',           # High - create connection
            'Hate',                # Medium - handle carefully
            'Spam'                 # Low - filter out
        ]


if __name__ == "__main__":
    # Test response templates
    rt = ResponseTemplates()
    
    print("="*60)
    print("RESPONSE TEMPLATES TEST")
    print("="*60)
    
    categories = ['Praise', 'Constructive Criticism', 'Threat', 'Question']
    
    for category in categories:
        print(f"\n{category}:")
        print("-" * 40)
        responses = rt.get_all_responses(category)
        for i, response in enumerate(responses, 1):
            print(f"{i}. {response}")
    
    print("\n" + "="*60)
    print("ACTION GUIDELINES")
    print("="*60)
    
    guide = rt.get_action_guide()
    for category, info in guide.items():
        print(f"\n{category}:")
        print(f"  Priority: {info['priority']}")
        print(f"  Action: {info['action']}")
        print(f"  Notes: {info['notes']}")

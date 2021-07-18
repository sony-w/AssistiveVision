class DescriptionRequest:
    """
    Parses a request for the description of a tweet, does basic validation to
    ensure that the request is for captioning and refers to a good target.
    """
    def __init__(self, hook_event):
        if hook_event['for_user_id'] != '1407099013292306439':
            self._set_invalid(f'message for wrong user ({hook_event["for_user_id"]})')
            return
        
        if not 'tweet_create_events' in hook_event:
            self._set_invalid('no tweet_create_events present')
            return
    
        create_events = hook_event['tweet_create_events']
        if len(create_events) != 1:
            self._set_invalid(f'not sure how to handle {len(create_events)} tweet_create_events')
            return
    
        create_event = create_events[0]
        self.tweet_id = create_event['id']
    
        if not '@helpertext desc' in create_event['text'].lower():
            self._set_invalid(f'text is not a valid description request: {create_event["text"]}')
            return
        if not 'in_reply_to_status_id' in create_event or create_event['in_reply_to_status_id'] is None:
            self._set_invalid(f'tweet ({self.tweet_id}) is not a reply')
            return
        
        self.target_tweet_id = create_event['in_reply_to_status_id']
        
        self.is_valid = True
        return
        
    def _set_invalid(self, msg):
        self.is_valid = False
        self.message = msg
class ReplyPoster:
    """
    Posts replies to tweets from the bot. Will optionally truncate or break the
    reply across multiple tweets.
    """

    def __init__(self, api, max_reply_tweets: int, is_test: bool) -> None:
        self._api = api
        self._max_reply_tweets = max_reply_tweets
        self._max_tweet_len = 200
        self._is_test = is_test

    def _post_status(self, reply_to_id, msg):
        api = self._api
        if self._is_test:
            print('not posting status since this is a test')
            return type('',(object,),{"id": 13132})()
        else:
            print(f'reply to id {reply_to_id} with {msg}')
            status = api.update_status(
                status = msg, 
                in_reply_to_status_id = reply_to_id, 
                auto_populate_reply_metadata=True)
        
        return status

    def post_reply(self, reply_to_id, msg):
        print(f'replying to {reply_to_id} with: {msg}')

        # break the reply into chunks of max tweet length
        parts = [msg[i:i+self._max_tweet_len] for i in range(0, len(msg), self._max_tweet_len)]

        # limit the total number of replies to max reply tweets
        if len(parts) > self._max_reply_tweets:
            parts = parts[:self._max_reply_tweets]
            parts[-1] = parts[-1] + '...'

        # add count markers
        if len(parts) > 1:
            parts = [parts[i] + f' ({i+1} of {len(parts)})' for i in range(0, len(parts))]
        
        status_id = reply_to_id
        for cur in parts:
            status = self._post_status(status_id, cur)
            status_id = status.id
            
        return status_id
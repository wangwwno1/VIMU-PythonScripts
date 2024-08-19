from collections import OrderedDict

import genpy
import rospy


class BaseROSController:
    def __init__(self):
        self._messages = OrderedDict()
        self._services = OrderedDict()
        self._subscribe_topics = OrderedDict()
        self._publish_topics = OrderedDict()

    def __getattr__(self, name):
        if '_messages' in self.__dict__:
            messages = self.__dict__['_messages']
            if name in messages:
                return messages[name]
        if '_services' in self.__dict__:
            services = self.__dict__['_services']
            if name in services:
                return services[name]
        if '_subscribe_topics' in self.__dict__:
            sub_topics = self.__dict__['_subscribe_topics']
            if name in sub_topics:
                return sub_topics[name]
        if '_publish_topics' in self.__dict__:
            pub_topics = self.__dict__['_publish_topics']
            if name in pub_topics:
                return pub_topics[name]

        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, name))

    def __setattr__(self, name, value):
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        if isinstance(value, genpy.Message):
            messages = self.__dict__.get('_messages')
            if messages is None:
                raise AttributeError("cannot assign messages before BaseROSController.__init__() call")
            remove_from(self.__dict__, self._services, self._subscribe_topics, self._publish_topics)
            messages[name] = value
        elif isinstance(value, rospy.ServiceProxy):
            services = self.__dict__.get('_services')
            if services is None:
                raise AttributeError("cannot assign service before before BaseROSController.__init__() call")
            remove_from(self.__dict__, self._messages, self._subscribe_topics, self._publish_topics)
            services[name] = value
        elif isinstance(value, rospy.Subscriber):
            sub_topics = self.__dict__.get('_subscribe_topics')
            if sub_topics is None:
                raise AttributeError("cannot assign subscriber before before BaseROSController.__init__() call")
            remove_from(self.__dict__, self._messages, self._services, self._publish_topics)
            sub_topics[name] = value
        elif isinstance(value, rospy.Publisher):
            pub_topics = self.__dict__.get('_publish_topics')
            if pub_topics is None:
                raise AttributeError("cannot assign subscriber before before BaseROSController.__init__() call")
            remove_from(self.__dict__, self._messages, self._services, self._subscribe_topics)
            pub_topics[name] = value
        else:
            object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in self._messages:
            del self._messages[name]
        elif name in self._services:
            del self._services[name]
        elif name in self._subscribe_topics:
            del self._subscribe_topics[name]
        elif name in self._publish_topics:
            del self._publish_topics[name]
        else:
            object.__delattr__(self, name)

    def messages(self):
        for name, msg in self.named_messages():
            yield msg

    def named_messages(self):
        for name, msg in self._messages.items():
            yield name, msg

    def services(self):
        for name, srv in self.named_services():
            yield srv

    def named_services(self):
        for name, srv in self._services.items():
            yield name, srv

    def sub_topics(self):
        for name, sub_topic in self.named_sub_topics():
            yield sub_topic

    def named_sub_topics(self):
        for name, sub_topic in self._subscribe_topics.items():
            yield name, sub_topic

    def pub_topics(self):
        for name, pub_topic in self.named_pub_topics():
            yield pub_topic

    def named_pub_topics(self):
        for name, pub_topic in self._publish_topics.items():
            yield name, pub_topic

    def close(self):
        for srv in self.services():
            srv.close()
        for sub_topic in self.sub_topics():
            sub_topic.unregister()
        for pub_topic in self.pub_topics():
            pub_topic.unregister()
import asyncio
import inspect

import pyglet


class AsyncEventDispatcher(pyglet.event.EventDispatcher):
    async def dispatch_event(self, event_type, *args):
        "Dispatch a single event to the attached coroutine handlers.
        The event is propagated to all handlers from from the top of the stack
        until one returns `EVENT_HANDLED`.  This method should be used only by
        :py:class:`~pyglet.event.EventDispatcher` implementors; applications should call
        the ``dispatch_events`` method.
        Since pyglet 1.2, the method returns `EVENT_HANDLED` if an event
        handler returned `EVENT_HANDLED` or `EVENT_UNHANDLED` if all events
        returned `EVENT_UNHANDLED`.  If no matching event handlers are in the
        stack, ``False`` is returned.
        :Parameters:
            `event_type` : str
                Name of the event.
            `args` : sequence
                Arguments to pass to the event handler.
        :rtype: bool or None
        :return: (Since pyglet 1.2) `EVENT_HANDLED` if an event handler
            returned `EVENT_HANDLED`; `EVENT_UNHANDLED` if one or more event
            handlers were invoked but returned only `EVENT_UNHANDLED`;
            otherwise ``False``.  In pyglet 1.1 and earlier, the return value
            is always ``None``.
        """
        assert hasattr(self, 'event_types'), (
            "No events registered on this EventDispatcher. "
            "You need to register events with the class method "
            "EventDispatcher.register_event_type('event_name')."
        )
        assert event_type in self.event_types,\
            "%r not found in %r.event_types == %r" % (event_type, self, self.event_types)

        invoked = False

        # Search handler stack for matching event handlers
        for frame in list(self._event_stack):
            handler = frame.get(event_type, None)
            if not handler:
                continue
            if isinstance(handler, WeakMethod):
                handler = handler()
                assert handler is not None
            try:
                invoked = True
                if inspect.iscoroutinefunction(handle):
                    if await handler(*args):
                        return EVENT_HANDLED
                else:
                    if handler(*args):
                        return EVENT_HANDLED
            except TypeError as exception:
                self._raise_dispatch_exception(event_type, args, handler, exception)

        # Check instance for an event handler
        try:
            if getattr(self, event_type)(*args):
                return EVENT_HANDLED
        except AttributeError:
            pass
        except TypeError as exception:
            self._raise_dispatch_exception(event_type, args, getattr(self, event_type), exception)
        else:
            invoked = True

        if invoked:
            return EVENT_UNHANDLED

        return False


class AsyncWindow()

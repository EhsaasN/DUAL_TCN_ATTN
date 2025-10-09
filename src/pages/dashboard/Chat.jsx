import React, { useState, useRef, useEffect } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { setLoading, setToast } from '../../state/uiSlice';
import { api } from '../../api/api';

function Message({ msg, isUser }) {
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-2`}>
      <div className={`max-w-xs px-4 py-2 rounded-2xl shadow-md ${isUser ? 'bg-primary text-black' : 'bg-surface text-white'} ${isUser ? 'rounded-br-none' : 'rounded-bl-none'}`}>{msg}</div>
    </div>
  );
}

export default function Chat() {
  const [messages, setMessages] = useState([
    { user: false, text: 'Hi! I am your anomaly detection assistant.' },
  ]);
  const [input, setInput] = useState('');
  const dispatch = useDispatch();
  const user = useSelector((state) => state.auth.user);
  const chatRef = useRef(null);

  useEffect(() => {
    chatRef.current?.scrollTo(0, chatRef.current.scrollHeight);
  }, [messages]);

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;
    const msg = input;
    setMessages((msgs) => [...msgs, { user: true, text: msg }]);
    setInput('');
    dispatch(setLoading(true));
    try {
      const res = await api.chat({ message: msg, user: user?.name });
      setMessages((msgs) => [...msgs, { user: false, text: res.data.reply || '...' }]);
    } catch (err) {
      setMessages((msgs) => [...msgs, { user: false, text: 'Error: Could not get reply.' }]);
      dispatch(setToast({ type: 'error', message: 'Chat failed.' }));
    } finally {
      dispatch(setLoading(false));
    }
  };

  return (
    <div className="max-w-2xl mx-auto bg-card p-8 rounded-lg shadow-lg border border-border mt-8 flex flex-col h-[70vh]">
      <h2 className="text-2xl font-bold mb-4 text-center">Chat Assistant</h2>
      <div ref={chatRef} className="flex-1 overflow-y-auto mb-4 pr-2">
        {messages.map((m, i) => (
          <Message key={i} msg={m.text} isUser={m.user} />
        ))}
      </div>
      <form onSubmit={sendMessage} className="flex gap-2">
        <input
          className="flex-1 px-4 py-2 rounded-2xl bg-surface border border-border focus:outline-primary text-white"
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Type your message..."
        />
        <button type="submit" className="px-4 py-2 rounded-2xl bg-primary hover:bg-accent font-bold">Send</button>
      </form>
    </div>
  );
}

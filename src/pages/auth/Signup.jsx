import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useDispatch } from 'react-redux';
import { loginSuccess } from '../../state/authSlice';
import { setLoading, setToast } from '../../state/uiSlice';
import { api } from '../../api/api';

export default function Signup() {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const dispatch = useDispatch();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (password !== confirm) {
      dispatch(setToast({ type: 'error', message: 'Passwords do not match.' }));
      return;
    }
    dispatch(setLoading(true));
    try {
      const res = await api.signup({ name, email, password });
      dispatch(loginSuccess(res.data.token));
      dispatch(setToast({ type: 'success', message: 'Signup successful!' }));
      navigate('/dashboard/upload');
    } catch (err) {
      dispatch(setToast({ type: 'error', message: 'Signup failed.' }));
    } finally {
      dispatch(setLoading(false));
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-bgDark">
      <form onSubmit={handleSubmit} className="bg-card p-8 rounded-lg shadow-lg w-full max-w-sm border border-border">
        <h2 className="text-2xl font-bold mb-6 text-center">Signup</h2>
        <input type="text" required placeholder="Name" value={name} onChange={e => setName(e.target.value)} className="mb-4 w-full px-4 py-2 rounded bg-surface border border-border focus:outline-primary" />
        <input type="email" required placeholder="Email" value={email} onChange={e => setEmail(e.target.value)} className="mb-4 w-full px-4 py-2 rounded bg-surface border border-border focus:outline-primary" />
        <input type="password" required placeholder="Password" value={password} onChange={e => setPassword(e.target.value)} className="mb-4 w-full px-4 py-2 rounded bg-surface border border-border focus:outline-primary" />
        <input type="password" required placeholder="Confirm Password" value={confirm} onChange={e => setConfirm(e.target.value)} className="mb-6 w-full px-4 py-2 rounded bg-surface border border-border focus:outline-primary" />
        <button type="submit" className="w-full py-2 rounded bg-primary hover:bg-accent font-bold">Signup</button>
        <div className="flex justify-between mt-4 text-sm">
          <Link to="/login" className="text-accent hover:underline">Login</Link>
        </div>
      </form>
    </div>
  );
}

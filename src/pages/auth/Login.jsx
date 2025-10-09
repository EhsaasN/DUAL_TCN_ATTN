import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useDispatch } from 'react-redux';
import { loginSuccess } from '../../state/authSlice';
import { setLoading, setToast } from '../../state/uiSlice';
import { api } from '../../api/api';

export default function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const dispatch = useDispatch();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    dispatch(setLoading(true));
    try {
      const res = await api.login({ email, password });
      dispatch(loginSuccess(res.data.token));
      dispatch(setToast({ type: 'success', message: 'Login successful!' }));
      navigate('/dashboard/upload');
    } catch (err) {
      dispatch(setToast({ type: 'error', message: 'Login failed.' }));
    } finally {
      dispatch(setLoading(false));
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-bgDark">
      <form onSubmit={handleSubmit} className="bg-card p-8 rounded-lg shadow-lg w-full max-w-sm border border-border">
        <h2 className="text-2xl font-bold mb-6 text-center">Login</h2>
        <input type="email" required placeholder="Email" value={email} onChange={e => setEmail(e.target.value)} className="mb-4 w-full px-4 py-2 rounded bg-surface border border-border focus:outline-primary" />
        <input type="password" required placeholder="Password" value={password} onChange={e => setPassword(e.target.value)} className="mb-6 w-full px-4 py-2 rounded bg-surface border border-border focus:outline-primary" />
        <button type="submit" className="w-full py-2 rounded bg-primary hover:bg-accent font-bold">Login</button>
        <div className="flex justify-between mt-4 text-sm">
          <Link to="/signup" className="text-accent hover:underline">Signup</Link>
          <Link to="/forgot-password" className="text-accent hover:underline">Forgot Password?</Link>
        </div>
      </form>
    </div>
  );
}

import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { useDispatch } from 'react-redux';
import { setLoading, setToast } from '../../state/uiSlice';
import { api } from '../../api/api';

export default function ForgotPassword() {
  const [email, setEmail] = useState('');
  const [sent, setSent] = useState(false);
  const dispatch = useDispatch();

  const handleSubmit = async (e) => {
    e.preventDefault();
    dispatch(setLoading(true));
    try {
      await api.forgotPassword({ email });
      setSent(true);
      dispatch(setToast({ type: 'success', message: 'Password reset link sent.' }));
    } catch (err) {
      dispatch(setToast({ type: 'error', message: 'Failed to send reset link.' }));
    } finally {
      dispatch(setLoading(false));
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-bgDark">
      <form onSubmit={handleSubmit} className="bg-card p-8 rounded-lg shadow-lg w-full max-w-sm border border-border">
        <h2 className="text-2xl font-bold mb-6 text-center">Forgot Password</h2>
        {sent ? (
          <div className="text-primary mb-6">Check your email for a reset link.</div>
        ) : (
          <input type="email" required placeholder="Email" value={email} onChange={e => setEmail(e.target.value)} className="mb-6 w-full px-4 py-2 rounded bg-surface border border-border focus:outline-primary" />
        )}
        <button type="submit" className="w-full py-2 rounded bg-primary hover:bg-accent font-bold" disabled={sent}>
          {sent ? 'Sent' : 'Send Reset Link'}
        </button>
        <div className="flex justify-between mt-4 text-sm">
          <Link to="/login" className="text-accent hover:underline">Login</Link>
          <Link to="/signup" className="text-accent hover:underline">Signup</Link>
        </div>
      </form>
    </div>
  );
}

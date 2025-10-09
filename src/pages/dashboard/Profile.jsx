import React, { useState } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { updateUser } from '../../state/authSlice';
import { setToast } from '../../state/uiSlice';

export default function Profile() {
  const user = useSelector((state) => state.auth.user);
  const [name, setName] = useState(user?.name || '');
  const [email, setEmail] = useState(user?.email || '');
  const [password, setPassword] = useState('');
  const dispatch = useDispatch();

  const handleUpdate = (e) => {
    e.preventDefault();
    // Dummy update, replace with backend call
    dispatch(updateUser({ name, email }));
    dispatch(setToast({ type: 'success', message: 'Profile updated!' }));
  };

  const handlePassword = (e) => {
    e.preventDefault();
    // Dummy update, replace with backend call
    if (!password) return;
    dispatch(setToast({ type: 'success', message: 'Password changed!' }));
    setPassword('');
  };

  return (
    <div className="max-w-lg mx-auto bg-card p-8 rounded-lg shadow-lg border border-border mt-8">
      <h2 className="text-2xl font-bold mb-6 text-center">Profile</h2>
      <form onSubmit={handleUpdate} className="flex flex-col gap-4 mb-8">
        <input
          type="text"
          value={name}
          onChange={e => setName(e.target.value)}
          className="w-full px-4 py-2 rounded bg-surface border border-border text-white"
          placeholder="Name"
        />
        <input
          type="email"
          value={email}
          onChange={e => setEmail(e.target.value)}
          className="w-full px-4 py-2 rounded bg-surface border border-border text-white"
          placeholder="Email"
        />
        <button type="submit" className="py-2 rounded bg-primary hover:bg-accent font-bold">Update Info</button>
      </form>
      <form onSubmit={handlePassword} className="flex flex-col gap-4">
        <input
          type="password"
          value={password}
          onChange={e => setPassword(e.target.value)}
          className="w-full px-4 py-2 rounded bg-surface border border-border text-white"
          placeholder="New Password"
        />
        <button type="submit" className="py-2 rounded bg-accent hover:bg-primary font-bold">Change Password</button>
      </form>
    </div>
  );
}

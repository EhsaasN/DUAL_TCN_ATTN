import { createSlice } from '@reduxjs/toolkit';
import { jwtDecode } from 'jwt-decode';

const token = localStorage.getItem('token');
let user = null;
try {
  user = token ? jwtDecode(token) : null;
} catch (err) {
  localStorage.removeItem('token');
}

const initialState = {
  token: token || null,
  user: user || null,
  isAuthenticated: !!token,
};

const authSlice = createSlice({
  name: 'auth',
  initialState,
  reducers: {
    loginSuccess(state, action) {
      state.token = action.payload;
      state.user = jwtDecode(action.payload);
      state.isAuthenticated = true;
      localStorage.setItem('token', action.payload);
    },
    logout(state) {
      state.token = null;
      state.user = null;
      state.isAuthenticated = false;
      localStorage.removeItem('token');
    },
    updateUser(state, action) {
      state.user = { ...state.user, ...action.payload };
    },
  },
});

export const { loginSuccess, logout, updateUser } = authSlice.actions;
export default authSlice.reducer;

import { createSlice } from '@reduxjs/toolkit';

const initialState = {
  loading: false,
  toast: null,
  theme: 'dark',
  error: null,
};

const uiSlice = createSlice({
  name: 'ui',
  initialState,
  reducers: {
    setLoading(state, action) {
      state.loading = action.payload;
    },
    setToast(state, action) {
      state.toast = action.payload;
    },
    setTheme(state, action) {
      state.theme = action.payload;
    },
    setError(state, action) {
      state.error = action.payload;
    },
    clearError(state) {
      state.error = null;
    },
  },
});

export const { setLoading, setToast, setTheme, setError, clearError } = uiSlice.actions;
export default uiSlice.reducer;

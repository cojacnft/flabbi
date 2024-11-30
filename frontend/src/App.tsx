import React, { useState } from 'react';
import {
  ThemeProvider,
  createTheme,
  CssBaseline,
  Box,
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Container,
  useMediaQuery,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Dashboard as DashboardIcon,
  Analytics as AnalyticsIcon,
  Settings as SettingsIcon,
  Token as TokenIcon,
  Brightness4 as DarkModeIcon,
  Brightness7 as LightModeIcon,
} from '@mui/icons-material';
import { Provider } from 'react-redux';
import { store } from './store';
import Dashboard from './components/Dashboard';
import AnalyticsPanel from './components/Analytics/AnalyticsPanel';
import SettingsPanel from './components/Settings/SettingsPanel';
import TokenManagement from './components/Tokens/TokenManagement';

const App: React.FC = () => {
  const [darkMode, setDarkMode] = useState(true);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [currentView, setCurrentView] = useState<string>('dashboard');
  const isMobile = useMediaQuery('(max-width:600px)');

  const theme = createTheme({
    palette: {
      mode: darkMode ? 'dark' : 'light',
      primary: {
        main: darkMode ? '#90caf9' : '#1976d2',
      },
      secondary: {
        main: darkMode ? '#f48fb1' : '#dc004e',
      },
      background: {
        default: darkMode ? '#0a1929' : '#f5f5f5',
        paper: darkMode ? '#132f4c' : '#ffffff',
      },
    },
    components: {
      MuiPaper: {
        styleOverrides: {
          root: {
            backgroundImage: 'none',
          },
        },
      },
    },
  });

  const toggleDrawer = () => {
    setDrawerOpen(!drawerOpen);
  };

  const handleNavigation = (view: string) => {
    setCurrentView(view);
    if (isMobile) {
      setDrawerOpen(false);
    }
  };

  const drawerContent = (
    <Box sx={{ width: 250 }}>
      <List>
        <ListItem>
          <Typography variant="h6" sx={{ p: 2 }}>
            Flash Loan Bot
          </Typography>
        </ListItem>
        <Divider />
        <ListItem
          button
          selected={currentView === 'dashboard'}
          onClick={() => handleNavigation('dashboard')}
        >
          <ListItemIcon>
            <DashboardIcon />
          </ListItemIcon>
          <ListItemText primary="Dashboard" />
        </ListItem>
        <ListItem
          button
          selected={currentView === 'analytics'}
          onClick={() => handleNavigation('analytics')}
        >
          <ListItemIcon>
            <AnalyticsIcon />
          </ListItemIcon>
          <ListItemText primary="Analytics" />
        </ListItem>
        <ListItem
          button
          selected={currentView === 'tokens'}
          onClick={() => handleNavigation('tokens')}
        >
          <ListItemIcon>
            <TokenIcon />
          </ListItemIcon>
          <ListItemText primary="Tokens" />
        </ListItem>
        <ListItem
          button
          selected={currentView === 'settings'}
          onClick={() => handleNavigation('settings')}
        >
          <ListItemIcon>
            <SettingsIcon />
          </ListItemIcon>
          <ListItemText primary="Settings" />
        </ListItem>
      </List>
    </Box>
  );

  const renderContent = () => {
    switch (currentView) {
      case 'dashboard':
        return <Dashboard />;
      case 'analytics':
        return <AnalyticsPanel />;
      case 'tokens':
        return <TokenManagement />;
      case 'settings':
        return <SettingsPanel />;
      default:
        return <Dashboard />;
    }
  };

  return (
    <Provider store={store}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Box sx={{ display: 'flex' }}>
          <AppBar
            position="fixed"
            sx={{
              zIndex: (theme) => theme.zIndex.drawer + 1,
            }}
          >
            <Toolbar>
              <IconButton
                color="inherit"
                edge="start"
                onClick={toggleDrawer}
                sx={{ mr: 2, display: { sm: 'none' } }}
              >
                <MenuIcon />
              </IconButton>
              <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
                Flash Loan Arbitrage
              </Typography>
              <IconButton
                color="inherit"
                onClick={() => setDarkMode(!darkMode)}
              >
                {darkMode ? <LightModeIcon /> : <DarkModeIcon />}
              </IconButton>
            </Toolbar>
          </AppBar>

          <Box
            component="nav"
            sx={{
              width: { sm: 250 },
              flexShrink: { sm: 0 },
            }}
          >
            {/* Mobile drawer */}
            <Drawer
              variant="temporary"
              open={drawerOpen}
              onClose={toggleDrawer}
              ModalProps={{
                keepMounted: true, // Better mobile performance
              }}
              sx={{
                display: { xs: 'block', sm: 'none' },
                '& .MuiDrawer-paper': {
                  width: 250,
                },
              }}
            >
              {drawerContent}
            </Drawer>

            {/* Desktop drawer */}
            <Drawer
              variant="permanent"
              sx={{
                display: { xs: 'none', sm: 'block' },
                '& .MuiDrawer-paper': {
                  width: 250,
                  boxSizing: 'border-box',
                },
              }}
              open
            >
              {drawerContent}
            </Drawer>
          </Box>

          <Box
            component="main"
            sx={{
              flexGrow: 1,
              p: 3,
              width: { sm: `calc(100% - 250px)` },
              mt: 8,
            }}
          >
            <Container maxWidth="xl">
              {renderContent()}
            </Container>
          </Box>
        </Box>
      </ThemeProvider>
    </Provider>
  );
};

export default App;
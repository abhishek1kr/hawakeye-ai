import { Component } from 'react';

/**
 * IMP-26: React Error Boundary
 * Catches rendering errors in child component tree and shows a graceful fallback
 * instead of crashing the entire UI.
 *
 * Usage:
 *   <ErrorBoundary fallback={<div>Chart failed to load</div>}>
 *     <SomeChart />
 *   </ErrorBoundary>
 */
class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, info) {
    console.error('[ErrorBoundary]', error, info.componentStack);
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback ?? (
        <div style={{
          padding: '16px',
          borderRadius: '8px',
          border: '1px solid rgba(255,0,85,0.4)',
          background: 'rgba(255,0,85,0.08)',
          color: 'var(--text-muted)',
          fontSize: '0.85rem',
          textAlign: 'center',
        }}>
          ⚠️ Component failed to render.{' '}
          <span
            style={{ cursor: 'pointer', color: 'var(--accent-primary)', textDecoration: 'underline' }}
            onClick={() => this.setState({ hasError: false, error: null })}
          >
            Retry
          </span>
        </div>
      );
    }
    return this.props.children;
  }
}

export default ErrorBoundary;

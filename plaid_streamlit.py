import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import hashlib
import json

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Plaid Trust Layer for Agentic Finance",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #6B7280;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .approved { background: #D1FAE5; border-left: 4px solid #059669; padding: 1rem; border-radius: 4px; }
    .stepup { background: #FEF3C7; border-left: 4px solid #D97706; padding: 1rem; border-radius: 4px; }
    .denied { background: #FEE2E2; border-left: 4px solid #DC2626; padding: 1rem; border-radius: 4px; }
    .insight-box {
        background: #EFF6FF;
        border-left: 4px solid #2563EB;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    .signal-positive { color: #059669; }
    .signal-negative { color: #DC2626; }
    .signal-neutral { color: #D97706; }
    .code-block {
        background: #1F2937;
        color: #F9FAFB;
        padding: 1rem;
        border-radius: 8px;
        font-family: monospace;
        font-size: 0.85rem;
        overflow-x: auto;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SIMULATED DATA MODELS
# ============================================================================

# Simulated user database with risk profiles
USERS = {
    "user_alice_normal": {
        "name": "Alice Chen",
        "email": "alice@example.com",
        "account_age_days": 847,
        "linked_apps": 3,
        "apps_linked_7d": 0,
        "fraud_reports": 0,
        "breach_exposure": False,
        "device_changes_30d": 1,
        "avg_monthly_transactions": 45,
        "identity_verified": True,
        "verification_method": "document_selfie",
    },
    "user_bob_new": {
        "name": "Bob Martinez",
        "email": "bob@example.com",
        "account_age_days": 12,
        "linked_apps": 1,
        "apps_linked_7d": 1,
        "fraud_reports": 0,
        "breach_exposure": False,
        "device_changes_30d": 2,
        "avg_monthly_transactions": 8,
        "identity_verified": True,
        "verification_method": "document_only",
    },
    "user_charlie_risky": {
        "name": "Charlie Davis",
        "email": "charlie@example.com",
        "account_age_days": 45,
        "linked_apps": 7,
        "apps_linked_7d": 4,  # This triggers the 65x risk signal
        "fraud_reports": 0,
        "breach_exposure": True,
        "device_changes_30d": 5,
        "avg_monthly_transactions": 12,
        "identity_verified": True,
        "verification_method": "document_selfie",
    },
    "user_diana_flagged": {
        "name": "Diana Smith",
        "email": "diana@example.com",
        "account_age_days": 234,
        "linked_apps": 4,
        "apps_linked_7d": 0,
        "fraud_reports": 2,  # Has fraud reports in Beacon network
        "breach_exposure": True,
        "device_changes_30d": 0,
        "avg_monthly_transactions": 32,
        "identity_verified": True,
        "verification_method": "document_selfie",
    },
}

# Simulated registered agents
AGENTS = {
    "agent_billpay_acme": {
        "name": "ACME Bill Pay Assistant",
        "developer": "ACME Fintech Inc.",
        "registered_date": "2025-03-15",
        "permissions": ["read_balance", "initiate_ach"],
        "max_transaction": 1000,
        "transaction_count_30d": 4521,
        "fraud_rate_30d": 0.02,  # 0.02% fraud rate
        "verified": True,
    },
    "agent_invest_beta": {
        "name": "Beta Investment Rebalancer",
        "developer": "Beta Capital",
        "registered_date": "2025-06-01",
        "permissions": ["read_balance", "read_transactions", "initiate_ach"],
        "max_transaction": 5000,
        "transaction_count_30d": 892,
        "fraud_rate_30d": 0.05,
        "verified": True,
    },
    "agent_unknown": {
        "name": "Unknown Agent",
        "developer": "Unknown",
        "registered_date": "2025-01-20",
        "permissions": ["read_balance"],
        "max_transaction": 100,
        "transaction_count_30d": 23,
        "fraud_rate_30d": 2.1,  # High fraud rate - suspicious
        "verified": False,
    },
}

# ============================================================================
# TRUST SCORING ENGINE
# ============================================================================

def calculate_trust_signals(user_id, agent_id, transaction_amount):
    """
    Calculate individual trust signals and aggregate score.
    This simulates what Plaid's Trust Index API would return.
    """
    user = USERS.get(user_id, USERS["user_alice_normal"])
    agent = AGENTS.get(agent_id, AGENTS["agent_billpay_acme"])
    
    signals = {}
    
    # Signal 1: Account Age (max 15 points)
    if user["account_age_days"] > 365:
        signals["account_age"] = {"score": 15, "value": f"{user['account_age_days']} days", "status": "strong"}
    elif user["account_age_days"] > 90:
        signals["account_age"] = {"score": 10, "value": f"{user['account_age_days']} days", "status": "moderate"}
    elif user["account_age_days"] > 30:
        signals["account_age"] = {"score": 5, "value": f"{user['account_age_days']} days", "status": "weak"}
    else:
        signals["account_age"] = {"score": 0, "value": f"{user['account_age_days']} days", "status": "risk"}
    
    # Signal 2: Cross-Platform Velocity - THE 65x SIGNAL (max 20 points, can go negative)
    if user["apps_linked_7d"] >= 3:
        signals["cross_platform_velocity"] = {
            "score": -20,  # NEGATIVE - this is the 65x risk signal
            "value": f"{user['apps_linked_7d']} apps in 7 days",
            "status": "critical_risk",
            "insight": "65x higher fraud probability (Beacon network signal)"
        }
    elif user["apps_linked_7d"] >= 1:
        signals["cross_platform_velocity"] = {
            "score": 5,
            "value": f"{user['apps_linked_7d']} app in 7 days",
            "status": "moderate"
        }
    else:
        signals["cross_platform_velocity"] = {
            "score": 20,
            "value": "No recent app linking",
            "status": "strong"
        }
    
    # Signal 3: Beacon Fraud Reports (max 20 points)
    if user["fraud_reports"] == 0:
        signals["beacon_fraud_reports"] = {"score": 20, "value": "0 reports", "status": "strong"}
    elif user["fraud_reports"] == 1:
        signals["beacon_fraud_reports"] = {"score": 5, "value": "1 report", "status": "weak"}
    else:
        signals["beacon_fraud_reports"] = {"score": -15, "value": f"{user['fraud_reports']} reports", "status": "critical_risk"}
    
    # Signal 4: Breach Exposure (max 10 points)
    if not user["breach_exposure"]:
        signals["breach_exposure"] = {"score": 10, "value": "Not exposed", "status": "strong"}
    else:
        signals["breach_exposure"] = {"score": -5, "value": "PII found in breach", "status": "risk"}
    
    # Signal 5: Identity Verification (max 15 points)
    if user["identity_verified"] and user["verification_method"] == "document_selfie":
        signals["identity_verification"] = {"score": 15, "value": "Doc + Selfie verified", "status": "strong"}
    elif user["identity_verified"]:
        signals["identity_verification"] = {"score": 10, "value": "Document verified", "status": "moderate"}
    else:
        signals["identity_verification"] = {"score": 0, "value": "Not verified", "status": "risk"}
    
    # Signal 6: Device Consistency (max 10 points)
    if user["device_changes_30d"] <= 1:
        signals["device_consistency"] = {"score": 10, "value": f"{user['device_changes_30d']} changes", "status": "strong"}
    elif user["device_changes_30d"] <= 3:
        signals["device_consistency"] = {"score": 5, "value": f"{user['device_changes_30d']} changes", "status": "moderate"}
    else:
        signals["device_consistency"] = {"score": -5, "value": f"{user['device_changes_30d']} changes", "status": "risk"}
    
    # Signal 7: Agent Reputation (max 10 points)
    if agent["verified"] and agent["fraud_rate_30d"] < 0.1:
        signals["agent_reputation"] = {"score": 10, "value": f"{agent['fraud_rate_30d']}% fraud rate", "status": "strong"}
    elif agent["verified"]:
        signals["agent_reputation"] = {"score": 5, "value": f"{agent['fraud_rate_30d']}% fraud rate", "status": "moderate"}
    else:
        signals["agent_reputation"] = {"score": -10, "value": "Unverified agent", "status": "critical_risk"}
    
    # Signal 8: Transaction Amount vs History (max 10 points)
    typical_max = user["avg_monthly_transactions"] * 50  # rough heuristic
    if transaction_amount <= typical_max * 0.5:
        signals["transaction_pattern"] = {"score": 10, "value": "Within normal range", "status": "strong"}
    elif transaction_amount <= typical_max:
        signals["transaction_pattern"] = {"score": 5, "value": "At upper range", "status": "moderate"}
    else:
        signals["transaction_pattern"] = {"score": -5, "value": "Unusual amount", "status": "risk"}
    
    # Calculate aggregate score
    total_score = sum(s["score"] for s in signals.values())
    max_possible = 110  # Sum of all max positive scores
    normalized_score = max(0, min(100, (total_score / max_possible) * 100 + 30))  # Normalize to 0-100
    
    return {
        "signals": signals,
        "raw_score": total_score,
        "normalized_score": round(normalized_score, 1),
    }


def get_authorization_decision(trust_result, transaction_amount, agent_max):
    """
    Determine authorization decision based on trust score and transaction context.
    """
    score = trust_result["normalized_score"]
    signals = trust_result["signals"]
    
    # Check for critical risk signals that override score
    critical_risks = [s for s in signals.values() if s.get("status") == "critical_risk"]
    
    if critical_risks:
        if any("65x" in s.get("insight", "") for s in critical_risks):
            return {
                "decision": "STEP_UP_REQUIRED",
                "reason": "Cross-platform velocity indicates 65x elevated risk. User confirmation required.",
                "action": "Require 2FA + explicit user approval for this transaction",
                "color": "stepup"
            }
        elif any("fraud" in s.get("value", "").lower() for s in critical_risks):
            return {
                "decision": "DENIED",
                "reason": "User has fraud reports in Beacon consortium network.",
                "action": "Block transaction. Recommend manual review.",
                "color": "denied"
            }
        elif any("Unverified" in s.get("value", "") for s in critical_risks):
            return {
                "decision": "DENIED",
                "reason": "Agent is not verified in Plaid's agent registry.",
                "action": "Block transaction. Agent must complete verification.",
                "color": "denied"
            }
    
    # Score-based decisions
    if score >= 75:
        return {
            "decision": "APPROVED",
            "reason": f"High trust score ({score}/100). All signals within acceptable range.",
            "action": "Proceed with transaction. Log for audit.",
            "color": "approved"
        }
    elif score >= 50:
        if transaction_amount > 500:
            return {
                "decision": "STEP_UP_REQUIRED",
                "reason": f"Medium trust score ({score}/100) with transaction >${transaction_amount}.",
                "action": "Require user confirmation via push notification.",
                "color": "stepup"
            }
        else:
            return {
                "decision": "APPROVED",
                "reason": f"Medium trust score ({score}/100) but low transaction amount.",
                "action": "Proceed with enhanced logging.",
                "color": "approved"
            }
    else:
        return {
            "decision": "DENIED",
            "reason": f"Low trust score ({score}/100). Multiple risk signals detected.",
            "action": "Block transaction. Require full re-authentication.",
            "color": "denied"
        }


# ============================================================================
# MAIN APP
# ============================================================================

st.markdown('<p class="main-header">🔐 Plaid Trust Layer for Agentic Finance</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Demonstrating how Plaid\'s Identity + Beacon + Transfer stack enables safe autonomous AI agent transactions</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("## Configuration")
st.sidebar.markdown("---")

# Tab navigation
tab1, tab2, tab3, tab4 = st.tabs([
    "🔑 Agent Authorization",
    "📊 Trust Signal Deep-Dive", 
    "🧪 The 65x Problem",
    "🏗️ Architecture"
])

# ============================================================================
# TAB 1: AGENT AUTHORIZATION FLOW
# ============================================================================
with tab1:
    st.markdown("## Agent Transaction Authorization")
    st.markdown("Simulate an AI agent requesting to perform a financial action on behalf of a user.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Agent Request")
        
        selected_agent = st.selectbox(
            "AI Agent",
            options=list(AGENTS.keys()),
            format_func=lambda x: f"{AGENTS[x]['name']} ({AGENTS[x]['developer']})"
        )
        
        selected_user = st.selectbox(
            "User Account",
            options=list(USERS.keys()),
            format_func=lambda x: f"{USERS[x]['name']} - {USERS[x]['email']}"
        )
        
        action_type = st.selectbox(
            "Requested Action",
            ["Pay utility bill", "Transfer to savings", "Pay credit card", "Send to external account"]
        )
        
        transaction_amount = st.slider(
            "Transaction Amount ($)",
            min_value=10,
            max_value=2000,
            value=150,
            step=10
        )
        
        st.markdown("---")
        authorize_btn = st.button("🚀 Submit for Authorization", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("### 🛡️ Trust Layer Response")
        
        if authorize_btn:
            with st.spinner("Evaluating trust signals..."):
                import time
                time.sleep(1)
            
            # Calculate trust
            trust_result = calculate_trust_signals(selected_user, selected_agent, transaction_amount)
            decision = get_authorization_decision(trust_result, transaction_amount, AGENTS[selected_agent]["max_transaction"])
            
            # Display decision
            st.markdown(f"""
            <div class="{decision['color']}">
                <h3 style="margin:0;">{decision['decision']}</h3>
                <p style="margin:0.5rem 0;"><strong>Reason:</strong> {decision['reason']}</p>
                <p style="margin:0;"><strong>Action:</strong> {decision['action']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Trust score gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=trust_result["normalized_score"],
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': "#111827"},
                    'steps': [
                        {'range': [0, 50], 'color': "#FEE2E2"},
                        {'range': [50, 75], 'color': "#FEF3C7"},
                        {'range': [75, 100], 'color': "#D1FAE5"}
                    ],
                    'threshold': {
                        'line': {'color': "#111827", 'width': 4},
                        'thickness': 0.75,
                        'value': trust_result["normalized_score"]
                    }
                },
                title={'text': "Trust Score"}
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            # Signal breakdown
            st.markdown("**Signal Breakdown:**")
            for signal_name, signal_data in trust_result["signals"].items():
                status_emoji = "✅" if signal_data["status"] == "strong" else "⚠️" if signal_data["status"] in ["moderate", "weak"] else "🚨"
                score_color = "green" if signal_data["score"] > 0 else "red" if signal_data["score"] < 0 else "gray"
                st.markdown(f"{status_emoji} **{signal_name.replace('_', ' ').title()}**: {signal_data['value']} (:{score_color}[{signal_data['score']:+d} pts])")
                if "insight" in signal_data:
                    st.caption(f"   ⚡ {signal_data['insight']}")
        else:
            st.info("👈 Configure the agent request and click 'Submit for Authorization' to simulate the trust evaluation.")
            
            # Show what will be evaluated
            st.markdown("**Signals that will be evaluated:**")
            st.markdown("""
            1. Account age and history
            2. Cross-platform velocity (🔑 the 65x signal)
            3. Beacon fraud consortium reports
            4. Breach exposure check
            5. Identity verification level
            6. Device consistency
            7. Agent reputation score
            8. Transaction pattern analysis
            """)

# ============================================================================
# TAB 2: TRUST SIGNAL DEEP-DIVE
# ============================================================================
with tab2:
    st.markdown("## Trust Signal Analysis")
    st.markdown("Explore how different user profiles generate different trust scores.")
    
    # Compare all users
    st.markdown("### User Profile Comparison")
    
    comparison_data = []
    for user_id, user in USERS.items():
        trust = calculate_trust_signals(user_id, "agent_billpay_acme", 200)
        decision = get_authorization_decision(trust, 200, 1000)
        comparison_data.append({
            "User": user["name"],
            "Account Age": f"{user['account_age_days']}d",
            "Apps (7d)": user["apps_linked_7d"],
            "Fraud Reports": user["fraud_reports"],
            "Breach Exposed": "Yes" if user["breach_exposure"] else "No",
            "Trust Score": trust["normalized_score"],
            "Decision": decision["decision"]
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Color code the dataframe
    def color_decision(val):
        if val == "APPROVED":
            return "background-color: #D1FAE5"
        elif val == "STEP_UP_REQUIRED":
            return "background-color: #FEF3C7"
        else:
            return "background-color: #FEE2E2"
    
    styled_df = df.style.applymap(color_decision, subset=["Decision"])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Signal weights visualization
    st.markdown("### Signal Weight Distribution")
    
    signal_weights = pd.DataFrame({
        "Signal": [
            "Cross-Platform Velocity",
            "Beacon Fraud Reports", 
            "Account Age",
            "Identity Verification",
            "Breach Exposure",
            "Device Consistency",
            "Agent Reputation",
            "Transaction Pattern"
        ],
        "Max Points": [20, 20, 15, 15, 10, 10, 10, 10],
        "Category": [
            "Network Intelligence",
            "Network Intelligence",
            "User History",
            "Identity",
            "Network Intelligence",
            "Behavioral",
            "Agent",
            "Behavioral"
        ]
    })
    
    fig = px.bar(
        signal_weights,
        x="Signal",
        y="Max Points",
        color="Category",
        color_discrete_map={
            "Network Intelligence": "#0D9488",
            "User History": "#6366F1",
            "Identity": "#8B5CF6",
            "Behavioral": "#F59E0B",
            "Agent": "#EC4899"
        }
    )
    fig.update_layout(
        height=400,
        xaxis_tickangle=-45,
        margin=dict(b=100)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="insight-box">
        <strong>💡 PM Insight:</strong> Network Intelligence signals (Cross-Platform Velocity, Beacon Reports, Breach Exposure) 
        account for 50 of 110 possible points. This reflects Plaid's unique data moat—these signals are 
        <em>only possible</em> because Plaid sees behavior across 8,000+ fintechs and 500M+ accounts.
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# TAB 3: THE 65x PROBLEM
# ============================================================================
with tab3:
    st.markdown("## The 65x Problem in Agentic Finance")
    
    st.markdown("""
    Plaid's Beacon data shows that users who link 3+ apps from one account in 7 days have 
    **65x higher fraud probability**. This is a powerful signal—but it creates a problem for AI agents.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚨 The Problem")
        st.markdown("""
        A legitimate AI agent might:
        - Connect to a user's bank for balance checks
        - Link to a brokerage for investment data
        - Connect to a credit card for payment
        - Link to a savings account for transfers
        
        **That's 4 connections in rapid succession**—which looks exactly like fraud.
        """)
        
        # Visualization of the problem
        st.markdown("**Traditional Fraud Pattern:**")
        fraud_pattern = pd.DataFrame({
            "Day": [1, 2, 3, 4, 5, 6, 7],
            "Apps Linked": [1, 2, 1, 3, 0, 0, 0],
            "Type": ["Fraud"] * 7
        })
        
        st.markdown("**Legitimate Agent Pattern:**")
        agent_pattern = pd.DataFrame({
            "Day": [1, 2, 3, 4, 5, 6, 7],
            "Apps Linked": [4, 0, 0, 0, 0, 0, 0],
            "Type": ["Agent"] * 7
        })
    
    with col2:
        st.markdown("### ✅ The Solution")
        st.markdown("""
        Plaid's Trust Layer can distinguish agent behavior from fraud by:
        
        1. **Agent Identity Binding**: Agent is registered and verified with Plaid
        2. **Delegated Consent**: User explicitly authorized this agent
        3. **Behavioral Context**: Agent follows predictable patterns
        4. **Cross-Reference**: Agent's other users show similar benign patterns
        """)
        
        st.markdown("**Key Insight:**")
        st.markdown("""
        <div class="insight-box">
            Only Plaid can solve this because they see both:
            <ul>
                <li>The fraud patterns (from Beacon consortium)</li>
                <li>The agent registration (from their API)</li>
            </ul>
            A single institution can't distinguish—they just see "4 apps linked."
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Interactive demo
    st.markdown("### 🧪 Simulation: Same Behavior, Different Context")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Scenario A: Unknown Actor**")
        trust_unknown = calculate_trust_signals("user_charlie_risky", "agent_unknown", 500)
        decision_unknown = get_authorization_decision(trust_unknown, 500, 100)
        
        st.markdown(f"""
        <div class="{decision_unknown['color']}">
            <strong>{decision_unknown['decision']}</strong><br>
            Trust Score: {trust_unknown['normalized_score']}/100<br>
            {decision_unknown['reason']}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**Scenario B: Verified Agent**")
        # Simulate same user but with verified agent context
        st.markdown(f"""
        <div class="stepup">
            <strong>STEP_UP_REQUIRED</strong><br>
            Trust Score: 58/100<br>
            High velocity detected, but agent is verified. Requesting user confirmation.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    **The difference:** Same user behavior, but agent verification changes the risk calculus.
    Without the agent registry, both scenarios would be blocked.
    """)

# ============================================================================
# TAB 4: ARCHITECTURE
# ============================================================================
with tab4:
    st.markdown("## Trust Layer Architecture")
    
    st.markdown("""
    ### How Plaid's Existing Products Map to Agent Requirements
    """)
    
    # Architecture diagram as a table
    arch_data = pd.DataFrame({
        "Agent Need": [
            "Prove user authorized the agent",
            "Access account data",
            "Check if user is fraud risk",
            "Check if agent is legitimate",
            "Execute payment",
            "Log for compliance"
        ],
        "Plaid Product": [
            "Plaid Identity + Link",
            "Plaid Link (Transactions, Balance)",
            "Beacon + Trust Index",
            "🆕 Agent Registry (proposed)",
            "Plaid Transfer",
            "🆕 Agent Audit Log (proposed)"
        ],
        "Exists Today?": [
            "✅ Yes",
            "✅ Yes",
            "✅ Yes",
            "❌ New build",
            "✅ Yes",
            "❌ New build"
        ],
        "Effort": [
            "—",
            "—",
            "—",
            "Medium (3-6 months)",
            "—",
            "Low (1-2 months)"
        ]
    })
    
    st.dataframe(arch_data, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # API example
    st.markdown("### Proposed API: Agent Transaction Authorization")
    
    st.markdown("**Request:**")
    st.code("""
POST /v1/agent/authorize
{
  "agent_id": "agent_billpay_acme",
  "agent_signature": "eyJhbGciOiJSUzI1NiIs...",
  "user_token": "user-sandbox-abc123",
  "action": {
    "type": "ach_transfer",
    "amount": 150.00,
    "recipient": "Electric Company"
  }
}
    """, language="json")
    
    st.markdown("**Response:**")
    st.code("""
{
  "authorization_id": "auth_7xK9mN2pQ",
  "decision": "APPROVED",
  "trust_score": 82,
  "signals": {
    "cross_platform_velocity": {"score": 20, "status": "strong"},
    "beacon_fraud_reports": {"score": 20, "status": "strong"},
    "agent_reputation": {"score": 10, "status": "strong"},
    ...
  },
  "action_required": null,
  "expires_at": "2025-01-28T22:30:00Z"
}
    """, language="json")
    
    st.markdown("---")
    
    # Feature prioritization preview
    st.markdown("### Feature Prioritization (RICE)")
    
    rice_data = pd.DataFrame({
        "Feature": [
            "Agent Authentication API",
            "Real-time Trust Scoring (<100ms)",
            "Delegated Authorization Scopes",
            "Agent Audit Logging"
        ],
        "Reach": [9, 8, 8, 7],
        "Impact": [9, 9, 8, 7],
        "Confidence": [8, 7, 8, 9],
        "Effort": [7, 8, 7, 5],
        "RICE Score": [92.6, 63.0, 73.1, 88.2]
    })
    
    rice_data = rice_data.sort_values("RICE Score", ascending=False)
    
    fig = px.bar(
        rice_data,
        x="Feature",
        y="RICE Score",
        color="RICE Score",
        color_continuous_scale=["#FEE2E2", "#FEF3C7", "#D1FAE5"]
    )
    fig.update_layout(height=350, xaxis_tickangle=-15)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="insight-box">
        <strong>💡 Recommendation:</strong> Start with Agent Authentication API (highest RICE, foundational) 
        and Agent Audit Logging (high RICE, low effort). These establish the trust infrastructure 
        on which real-time scoring and scopes can build.
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6B7280; font-size: 0.85rem;'>
    <p><strong>PM Portfolio Project</strong> | Built to demonstrate product thinking for Plaid's agentic finance opportunity</p>
    <p>Sources: <a href='https://www.acquired.fm/episodes/undoing-a-5-billion-acquisition-and-building-a-durable-standalone-plaid-with-plaid-ceo-zach-perret'>Acquired Podcast (Zach Perret)</a> • 
    <a href='https://www.pymnts.com/data/2025/plaid-ceo-says-its-next-five-years-will-look-a-lot-different-than-the-last-five/'>PYMNTS Interview</a> • 
    <a href='https://plaid.com/products/beacon/'>Plaid Beacon Docs</a></p>
    <p><em>This is a demonstration prototype. Not affiliated with Plaid.</em></p>
</div>
""", unsafe_allow_html=True)

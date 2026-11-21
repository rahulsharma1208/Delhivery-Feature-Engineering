# Delhivery - Feature Engineering & Prediction Analysis

## 📚 Overview

This project focuses on feature engineering for predictive modeling in logistics and delivery analytics. The analysis explores customer behavior patterns across the last 45 months to build accurate churn prediction models using logistic regression techniques.

---

### **Business Problem**

- Understand driver attrition patterns at Delhivery
- Identify key features predicting customer churn
- Build predictive models to retain high-value customers

---

### **Analysis Approach**

**Key Insights:**
- 📊 **Dataset**: 40 drivers over 45 months (1,800 records)
- 🔍 **Customer Segmentation**: 9 distinct classes based on behavior patterns
- 🎯 **Goal**: Predict probability of customer churn using logistic regression

---

### **Methodology Highlights**

| Aspect | Analysis | Result |
|--------|----------|--------|
| Sample Size | 1,800 records | 45 months × 40 drivers |
| Customer Classes | 9 classes | Segmentation analysis |
| Method | Multi-class logistic regression | Predictive modeling |

---

### **Customer Classification**

The 9 customer classes identified based on behavior:
- **High**: High retention, high spending
- **Medium**: Balanced retention and engagement
- **Low**: Low engagement or churn probability

---

### **Technical Implementation**

```python
# Feature engineering for multi-class logistic regression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline

def build_feature_engineer(df):
    # Engineer features from raw data
    feature_select = SelectKBest(score_func=f_classif, k=10)
    
    pipeline = Pipeline([
        ('feature_transform', feature_select),
        ('clf', LogisticRegression())
    ])
    
    return pipeline.fit_transform(X_train.drop(columns=['class']))
```

---

### **Strategic Recommendations**

✅ **Segment-Based Retention**: Tailor strategies for each of the 9 customer classes  
✅ **Feature Importance**: Focus on predictive features for churn probability  
✅ **Predictive Model**: Deploy logistic regression system for real-time churn prediction  

---

### **Business Impact**

Understanding customer segmentation and applying feature engineering enables Delhivery to:
- Implement targeted retention programs for at-risk customers
- Optimize resource allocation based on predicted churn risk
- Develop data-driven strategies for customer lifecycle management

---

*By Rahul Sharma | Senior Business Analyst with expertise in feature engineering and prediction modeling*

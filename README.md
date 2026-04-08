# Rossmann Store Sales: Staffing Project

This project looks at daily store sales and uses machine learning to decide how many staff are needed each day.

## What the project does

- Predicts daily sales using past data  
- Uses uncertainty in predictions to adjust staffing  
- Adds extra staff on days where predictions are less certain  
- Compares this approach to a fixed staffing schedule  

## Models used

- Linear Regression (baseline)  
- Quantile Gradient Boosting (main model with prediction intervals)  

## Key idea

Instead of scheduling the same number of staff every day, this project adjusts staffing based on both predicted sales and how confident the model is.

If the model is unsure, it adds one extra staff member as a buffer.

## Results

- The model gives more flexible staffing decisions  
- It avoids overstaffing and understaffing  
- It reduces overall cost compared to a fixed schedule  

## Files

- `MSE433_Rossmann_Final_Project.ipynb` → main notebook with all code and charts  


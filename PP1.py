import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "product_details.csv"  # Replace with your path if needed
df = pd.read_csv(file_path)

# Step 1: Convert 'Date' to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')

# Step 2: Normalize negative values
df['Refunded Item Count'] = df['Refunded Item Count'].abs()

# Step 3: Create Return_Flag column
df['Return_Flag'] = ((df['Refunds'] < 0) | (df['Refunded Item Count'] > 0)).astype(int)

# Step 4: Remove duplicate rows
df = df.drop_duplicates()

# Step 5: Strip leading/trailing spaces in all string columns
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()

# Step 6: Keep only relevant columns
columns_to_keep = [
    'Item Name', 'Category', 'Date', 'Final Quantity', 'Total Revenue',
    'Refunds', 'Final Revenue', 'Sales Tax', 'Overall Revenue',
    'Refunded Item Count', 'Purchased Item Count', 'Return_Flag'
]
df_cleaned = df[columns_to_keep]

# Optional: Save cleaned data
df_cleaned.to_csv("cleaned_product_details.csv", index=False)


# Load cleaned dataset (make sure this file is already cleaned as per previous steps)
df = pd.read_csv("cleaned_product_details.csv")

# --- Add Time Features ---
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.to_period('M')
df['Year'] = df['Date'].dt.year

# --- Overall Return Rate ---
overall_return_rate = df['Return_Flag'].mean()
print(f"Overall Return Rate: {overall_return_rate:.2%}")

# --- Return Rate by Category ---
category_return_rate = df.groupby('Category')['Return_Flag'].mean().sort_values(ascending=False)

# --- Monthly Return Rate Trend ---
monthly_return_rate = df.groupby('Month')['Return_Flag'].mean()

# --- Top 10 Returned Products ---
top_returned_products = df[df['Return_Flag'] == 1]['Item Name'].value_counts().head(10)

# --- Set up plots ---
fig, axs = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle('E-Commerce Return Analysis Dashboard', fontsize=18, fontweight='bold')

# Plot 1: Return Rate by Category
sns.barplot(x=category_return_rate.values, y=category_return_rate.index, ax=axs[0, 0], palette='magma')
axs[0, 0].set_title("Return Rate by Category")
axs[0, 0].set_xlabel("Return Rate")
axs[0, 0].set_ylabel("Category")

# Plot 2: Monthly Return Rate Trend
monthly_return_rate.plot(marker='o', ax=axs[0, 1])
axs[0, 1].set_title("Monthly Return Rate Trend")
axs[0, 1].set_xlabel("Month")
axs[0, 1].set_ylabel("Return Rate")
axs[0, 1].tick_params(axis='x', rotation=45)

# Plot 3: Top Returned Products
top_returned_products.plot(kind='barh', ax=axs[1, 0], color='salmon')
axs[1, 0].set_title("Top 10 Returned Products")
axs[1, 0].invert_yaxis()
axs[1, 0].set_xlabel("Return Count")

# Plot 4: Revenue distribution by Return Flag
sns.boxplot(data=df, x='Return_Flag', y='Overall Revenue', ax=axs[1, 1], palette='coolwarm')
axs[1, 1].set_title("Revenue Distribution by Return Flag")
axs[1, 1].set_xticklabels(['Not Returned', 'Returned'])

# Plot 5: Return Flag Count
sns.countplot(x='Return_Flag', data=df, ax=axs[2, 0], palette='Set2')
axs[2, 0].set_title("Return Flag Distribution")
axs[2, 0].set_xticklabels(['Not Returned', 'Returned'])
axs[2, 0].set_ylabel("Count")

# Hide empty plot
axs[2, 1].axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

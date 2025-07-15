#!/bin/bash


# Simple Interest Calculator


echo "Enter the principal amount:"

read principal


echo "Enter the rate of interest (in %):"

read rate


echo "Enter the time period (in years):"

read time


# Calculate simple interest: (P * R * T) / 100

simple_interest=$(( principal * rate * time / 100 ))


echo "Simple Interest is: $simple_interest"

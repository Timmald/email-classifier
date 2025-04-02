import csv
import email_to_vector
csv.field_size_limit(1000000000)

ham_path = '/Users/adamlubomirski/Documents/Coding Projects/Email Classifier/email-classifier/datasets/enron_spam/ham.csv'
spam_path = '/Users/adamlubomirski/Documents/Coding Projects/Email Classifier/email-classifier/datasets/enron_spam/spam.csv'
column_data = []


#just testing first 10, not sure if it was correct because the messages that it did output were kinda grammatically strange
def first_ten_ham(file_path):
        with open(file_path, newline="") as file:
            reader = csv.reader(file)
            for row in reader:
                column_data.append(row[0])
        count = 0
        for item in column_data:
            print(item)
            print("")
            count+= 1
            if(count > 10):
                return
            

def first_ten_spam(file_path):
        with open(file_path, newline="") as file:
            reader = csv.reader(file)
            for row in reader:
                if(len(row) == 0):
                     continue
                column_data.append(row[0])
        count = 0
        for item in column_data:
            print(item)
            print("")
            count+= 1
            if(count > 10):
                return


first_ten_spam(spam_path)
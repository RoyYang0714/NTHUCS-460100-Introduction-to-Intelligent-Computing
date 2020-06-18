import csv

with open('../GAN.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['image_id','label'])

    for i in range(1900):
        writer.writerow(['{}.jpg'.format(i+1),"0"])
    
    for i in range(1800):
        writer.writerow(['{}.jpg'.format(i+1),"1"])

    for i in range(1900):
        writer.writerow(['{}.jpg'.format(i+1),"2"])
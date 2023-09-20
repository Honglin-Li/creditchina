# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import shutil
import sys
import os
import webbrowser
import time


# +
class PDFCollector:
    """
    Class to download social credit rating of firms in PDF format
    
    Methods
    -------
    single_download(name)
        Download pdf for a single company and store pdfs at 'target_path'.
    create_bulks(names, bulk_size)
        Split list of company names in dictionary of bulks of size 'bulk_size'.
    moving_file(file_name)
    is_file_existed_in_target(name)
    single_bulk_download(names)   
        Download pdfs for a single bulk and store pdfs at 'target_path'.
    bulk_download(names, bulk_size)
        Loop over all bulks and download pdfs for all companies in names (if exist).
    cleanup()
    """
    
#     webbrowser.register("firefox", None, webbrowser.GenericBrowser(r"C:\Program Files\Mozilla Firefox\firefox.exe"))
    webbrowser.register("firefox", None, webbrowser.BackgroundBrowser(r"C:\Program Files\Mozilla Firefox\firefox.exe"))
    
    def __init__(self, target_path, download_path, wait):
        """
        Parameters
        ----------
        target_path : str
            specify the target path where PDFs will be stored
        download_path : str
            download_path: specify path where browser saves downloaded files (typically in the 'Downloads' folder)
        wait : int
            specify the number of seconds to wait for download to be finished.
            The more waiting time is granted, the smaller the risk to miss a PDF document.
            This comes, however, at the cost of execution time.
        """        
        self.target_path = target_path
        self.download_path = download_path
        self.wait = wait
    
    
    def single_download(self, name):
        """Download pdf for a single company and store pdfs at 'target_path'.

        Parameters
        ----------
        name : str
            company name
        """ 
        
        self.name = name
        url = f'https://public.creditchina.gov.cn/credit-check/pdf/download?companyName={name}&entityType=1&uuid=&tyshxydm='
        webbrowser.open(url)
        
        timeout = time.time() + self.wait
        while not os.path.exists(f'{self.download_path}\\{name}.pdf'):
            time.sleep(3)
            if time.time() > timeout:
                return print(f"{name}: timeout.")

        shutil.move(f"{self.download_path}\\{name}.pdf", f"{self.target_path}\\{name}.pdf")
        print(f"{name}: download finished.")
        
        
    def create_bulks(self, names, bulk_size):
        """Split list of company names in dictionary of bulks of size 'bulk_size'.
        This allows to download PDF bulks in seperate browser windows in order to prevent browser to crash.

        Parameters
        ----------
        name : str
            company name
        bulk_size : int
        """ 
        
        bulk_dict = {}
        for ind, i in enumerate(range(0, len(names), bulk_size), start=0): 
            bulk_dict[ind] = names[i:i + bulk_size]
        return bulk_dict
    
    def moving_file(self, file_name):
        # moving file
        shutil.move(f"{self.download_path}\\{file_name}.pdf", f"{self.target_path}\\{file_name}.pdf")
        # moving successfully
        assert self.is_file_existed_in_target(file_name)
        
    def is_file_existed_in_target(self, name):
        return os.path.exists(f'{self.target_path}\\{name}.pdf')
    
    def single_bulk_download(self, names):
        """ Download pdfs for a single bulk and store pdfs at 'target_path'.

        Parameters
        ----------
        names : list of string
            list of company name
        """ 
        browser = webbrowser.get('firefox')
        incomplete_files = []
        for name in names:
            if not self.is_file_existed_in_target(name): # not exist in target folder
                incomplete_files.append(name)
        print(f'downloading {len(incomplete_files)} files')
        
        urls = [f'https://public.creditchina.gov.cn/credit-check/pdf/download?companyName={name}&entityType=1&uuid=&tyshxydm=' for name in incomplete_files]
        print(f'downloading {len(urls)} files')
        [browser.open_new_tab(url) for url in urls]
        # for url in urls:
        #     browser.open_new_tab(url)
        # [browser.open_new_tab('https://www.creditchina.gov.cn/') for url in urls]
        
        
        time.sleep(self.wait)
        print('Copying')
        for name in incomplete_files:
            if os.path.exists(f'{self.download_path}\\{name}.pdf') & ~os.path.exists(f'{self.download_path}\\{name}.pdf.part'):
                self.moving_file(name)
            else:
                with open(self.target_path + '\\companies_not_found.txt', 'a', encoding='utf-8') as company_not_found:
                    company_not_found.write("%s\n" % str(name))
        
        
    def bulk_download(self, names, bulk_size):
        """sLoop over all bulks and download pdfs for all companies in names (if exist).
        Close browser of current bulk once waiting time ('wait') has been exceeded.

        Parameters
        ----------
        names : list of str
            list of company name
        bulk_size : int
        """ 
        bulk_dict = self.create_bulks(names, bulk_size)
        n_bulks = len(bulk_dict)
        
        for i in range(n_bulks) :
            self.single_bulk_download(bulk_dict[i])
            # os.system('TASKKILL /F /IM firefox.exe')
            os.system('taskkill /im firefox.exe /f')
            os.system('taskkill /im firefox.exe /f')
            # os.system("taskkill /im firefox.exe")
            # os.system("taskkill /im firefox.exe /f")
            print(f"Bulk {i}/{n_bulks} finished.", (i+1)/n_bulks*100, "%")
            
    def cleanup(self):
        """
        Collect company names whose download had not finished within the 'wait' time.
        """         
        temp = os.listdir(self.download_path)
        names = [i.removesuffix('.pdf.part') for i in temp if 'pdf.part' in i]
        names = list(set(names))
        pdfs = [i for i in temp if any(name for name in names if name in i)]

        # Move files to target_path in seperate folder
        if len(pdfs) > 0:
            if not os.path.exists(f"{self.target_path}\\abort"):
                os.makedirs(f"{self.target_path}\\abort")
            for pdf in pdfs:
                shutil.move(f"{self.download_path}\\{pdf}", f"{self.target_path}\\abort\\{pdf}")     
        return names
# -




#!/usr/bin/env python
# coding: utf-8



"""
cli

The module provides commend line interface for the project.

Contents:
    - python cli.py createdb
    - python cli.py analysis
    - python cli.py regdb (-y 1 -r True)
"""


import argparse



def _get_parser():
    """
    Set up a command line interface.
        
    Returns
    ----------
    ArgumentParser
    """
    parser = argparse.ArgumentParser(description='The CLI can setup data and perform analysis')
    
    subparsers = parser.add_subparsers(title='action', help='Action to perform')
    
    # Command 1: create final dataset

    datasetup_parser = subparsers.add_parser('createdb', 
                                             help='Create the final dataset from the extracted credit report and performance data')
    
    datasetup_parser.set_defaults(action=_createdb)
    
    # Command 2: credit data analysis
    analysis_parser = subparsers.add_parser('analysis', 
                                            help='Genarate basic analysis tables and figures for the credit dataset')
    
    analysis_parser.set_defaults(action=_analysis)
    
    
    # Command 3: regression data setup
    regdb_parser = subparsers.add_parser('regdb',
                                 help='Regression data setup')
    regdb_parser.add_argument(
        '--year_diff', '-y',
        type=int,
        default=1,
        help='The year difference to calculate firm performance growth'
    )
    
    regdb_parser.add_argument(
        '--with_relationship', '-r',
        type=bool,
        default=True,
        help='Gerenate data with relationships be or not'
    )
    regdb_parser.set_defaults(action=_regdb)
    
    return parser


def _regdb(args):
    """
    Regression data set up.
    
    Parameters
    ----------
    args.year_diff : int, default 1
        The year gap to calculate performance growth.
    args.with_relationship : bool, default False
        If the original data include relationship data.
    """
    print('seting up regression data..........')
    
    from src.data_mining.regression_data_setup import setup_regression_data, save_stata_data, show_corr
    
    # data setup
    df = setup_regression_data(year_diff=args.year_diff, with_relationship=args.with_relationship)
    #df = setup_regression_data(year_diff=args['year_diff'], with_relationship=args['with_relationship'])
    
    file_name = 'panel_cp.dta'
    
    if args.with_relationship:
        file_name = 'panel_cpr.dta'
        
    # save
    print('the regression data is saved under data/processed_data/performance_panel')
    save_stata_data(df, file_name)
    
    # create correlation matrix figures
    #show_corr(df)



def _createdb(args):
    """
    Create the final dataset.
    """
    from src.data_transformation.create_dataset import create_credit_dataset
    
    print('creating the credit dataset...')
    
    create_credit_dataset()
    
    print('check the final dataset in data/processed_data...')



def _analysis(args):
    """
    Create credit analysis.
    """
    from src.data_transformation.create_dataset import credit_analysis
    
    print('Generating analysis...')
    
    credit_analysis()
    
    print('check the analysis in data/stat...')




def main():
    parser = _get_parser()
    args = parser.parse_args()

    if not hasattr(args, 'action'):
        parser.print_help()
        parser.exit()

    args.action(args)


if __name__ == '__main__':
    main()


from ssi.settings import Settings
import logging
import argparse


def main(args):
    settings = Settings.load(args.settings_filename)


if __name__ == "main":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--settings-filename", type=str,
                        default="./settings.yaml", help="The filename of the settings file")
    args = parser.parse_args()

    main(args)

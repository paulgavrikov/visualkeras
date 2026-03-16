# ReadTheDocs Setup Guide

This guide explains how to set up visualkeras documentation on ReadTheDocs for automatic builds and hosting.

## Overview

ReadTheDocs (RTD) provides free documentation hosting with automatic builds on every push to your GitHub repository. No manual updates required after initial setup.

## Prerequisites

Before starting, ensure you have:

1. A GitHub repository (visualkeras is already on GitHub)
2. A ReadTheDocs account (free at <https://readthedocs.org>)
3. Documentation source files in the docs directory (already done)
4. .readthedocs.yml in the repository root (already done)

## Step-by-Step Setup

### 1. Create a ReadTheDocs Account

Visit <https://readthedocs.org> and sign up using your GitHub account. This allows RTD to access your repositories.

### 2. Import Your Project

After signing in to ReadTheDocs:

1. Click "Import a Project" or go to your dashboard
2. Click "Import Manually" (or connect GitHub if available)
3. Fill in the form with these details:
   - **Name**: visualkeras
   - **Repository URL**: <https://github.com/paulgavrikov/visualkeras>
   - **Repository Type**: Git
   - **Default Version**: camera-ready (or master as default)
   - **Default Branch**: camera-ready

4. Click "Create Project"

### 3. Configure Build Settings

RTD should automatically detect the .readthedocs.yml file. Verify the settings:

1. Go to your project's admin page
2. Click "Settings"
3. Under "Advanced Settings":
   - Documentation type: Sphinx HTML
   - Python configuration file: docs/source/conf.py
   - Webhook URL: (RTD generates this automatically)

### 4. Configure GitHub Webhook

For automatic builds on push:

1. On ReadTheDocs, go to project Admin > Integrations
2. Set up a GitHub webhook (RTD handles this automatically if connected)
3. Verify the webhook appears in GitHub repo Settings > Webhooks

### 5. Trigger First Build

1. Return to RTD project dashboard
2. Click "Build Version"
3. RTD should automatically build your documentation
4. View live documentation at: <https://visualkeras.readthedocs.io/>

## Configuration Details

### .readthedocs.yml Explained

The configuration file controls how RTD builds your docs:

```yaml
version: 2
build:
  os: ubuntu-22.04
  tools:
    python: "3.11"
```

Specifies the build environment. Python 3.11 ensures compatibility with modern packages.

```yaml
python:
  version: 3.11
  install:
    - requirements: docs/requirements.txt
    - method: pip
      path: .
```

Installs documentation and package dependencies. The second line installs the visualkeras package itself (needed for autodoc).

```yaml
formats:
  - pdf
  - epub
```

Generates PDF and EPUB versions alongside HTML (optional, can slow builds).

### Sphinx Configuration

The docs/source/conf.py file contains Sphinx settings that RTD uses. Key settings:

- **extensions**: Enable autodoc, Napoleon, and other plugins
- **html_theme**: Furo theme (modern design)
- **intersphinx_mapping**: Link to Python and NumPy docs

## Customizing Your Documentation

### Change the Project Name

In RTD project settings, update project metadata to set custom display names.

### Configure Search Behavior

In .readthedocs.yml, adjust search ranking:

```yaml
search:
  ranking:
    api/index: -1  # Lower priority for API
```

### Set Version Aliases

In RTD Admin > Settings, you can:

- Set "latest" to point to "master" or "camera-ready"
- Choose which versions to display
- Create redirects for moved pages

### Add Custom Domain

To use visualkeras.org instead of readthedocs.org:

1. In RTD project settings, add your domain
2. Update DNS records at your registrar
3. RTD handles SSL certificates automatically

## Troubleshooting

### Build Fails with Import Errors

If autodoc fails to import visualkeras:

1. Ensure the package is explicitly installed in .readthedocs.yml:

   ```yaml
   python:
     install:
       - method: pip
         path: .
   ```

2. Check dependencies list in docs/requirements.txt

3. View build logs in RTD for specific error messages

### Documentation Looks Wrong

If styling or links are broken:

1. Clear RTD cache: Project Admin > Versions > rebuild
2. Check theme configuration in docs/source/conf.py
3. Verify all references use absolute paths, not relative

### Slow Builds

If documentation takes too long to build:

1. Disable PDF/EPUB generation in .readthedocs.yml
2. Reduce number of documentation files
3. Move computationally expensive examples to gallery

### Custom Domain Not Working

If your custom domain shows RTD errors:

1. Verify CNAME record points to readthedocs.io
2. Wait up to 30 minutes for DNS propagation
3. In RTD, re-validate the domain

## Monitoring and Maintenance

### Review Build Status

1. Visit RTD project dashboard
2. Check "Builds" tab for history
3. Click on build to view logs

### Set Up Notifications

In RTD project settings:

1. Under "Notifications"
2. Add your email for build failures
3. Or integrate with Slack/Discord webhook

### Update on Release

When releasing a new version of visualkeras:

1. Tag a release on GitHub
2. RTD automatically builds for the new version
3. Users can view docs for multiple versions

## Advanced: Local Testing

Test documentation locally before pushing:

```bash
cd docs
pip install -r requirements.txt
sphinx-build -b html source build/html
python -m http.server 8000 --directory build/html
```

Then visit <http://localhost:8000> to review.

## Next Steps

After RTD is configured:

1. Add documentation badge to README.md
2. Update CONTRIBUTING.md with doc build instructions
3. Monitor builds for any deprecation warnings
4. Gather feedback on documentation quality

## Resources

- ReadTheDocs Docs: <https://docs.readthedocs.io/>
- Sphinx Documentation: <https://www.sphinx-doc.org/>
- Furo Theme: <https://pradyunsg.me/furo/>

## Support

For ReadTheDocs issues, check:

1. RTD build logs in your project dashboard
2. <https://docs.readthedocs.io/en/stable/> for platform docs
3. Support forum: <https://community.readthedocs.org/>

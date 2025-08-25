module Jekyll
  class ArticleCategoryPageGenerator < Generator
    safe true

    def generate(site)
      if site.layouts.key? 'category'
        # Collect all categories from both posts and articles
        all_categories = Set.new

        site.posts.docs.each do |post|
          post.data['categories'].each { |c| all_categories.add(c) } if post.data['categories']
        end

        site.collections['articles'].docs.each do |article|
          article.data['categories'].each { |c| all_categories.add(c) } if article.data['categories']
        end if site.collections['articles']

        # Generate category pages
        all_categories.each do |category|
          # Get posts and articles for this category
          posts = site.posts.docs.select { |post| post.data['categories'] && post.data['categories'].include?(category) }
          articles = site.collections['articles'] ? site.collections['articles'].docs.select { |article| article.data['categories'] && article.data['categories'].include?(category) } : []
          
          all_content = (posts + articles).sort_by { |item| item.data['date'] || Time.at(0) }.reverse

          site.pages << CategoryPage.new(site, site.source, category, all_content)
        end
      end
    end
  end

  class CategoryPage < Page
    def initialize(site, base, category, posts)
      @site = site
      @base = base
      @dir = File.join('categories', Utils.slugify(category))
      @name = 'index.html'

      self.process(@name)
      self.read_yaml(File.join(base, '_layouts'), 'category.html')
      self.data['category'] = category
      self.data['posts'] = posts
      self.data['title'] = "Category: #{category}"
    end
  end

  class ArticleTagPageGenerator < Generator
    safe true

    def generate(site)
      if site.layouts.key? 'tag'
        # Collect all tags from both posts and articles
        all_tags = Set.new

        site.posts.docs.each do |post|
          post.data['tags'].each { |t| all_tags.add(t) } if post.data['tags']
        end

        site.collections['articles'].docs.each do |article|
          article.data['tags'].each { |t| all_tags.add(t) } if article.data['tags']
        end if site.collections['articles']

        # Generate tag pages
        all_tags.each do |tag|
          # Get posts and articles for this tag
          posts = site.posts.docs.select { |post| post.data['tags'] && post.data['tags'].include?(tag) }
          articles = site.collections['articles'] ? site.collections['articles'].docs.select { |article| article.data['tags'] && article.data['tags'].include?(tag) } : []
          
          all_content = (posts + articles).sort_by { |item| item.data['date'] || Time.at(0) }.reverse

          site.pages << TagPage.new(site, site.source, tag, all_content)
        end
      end
    end
  end

  class TagPage < Page
    def initialize(site, base, tag, posts)
      @site = site
      @base = base
      @dir = File.join('tags', Utils.slugify(tag))
      @name = 'index.html'

      self.process(@name)
      self.read_yaml(File.join(base, '_layouts'), 'tag.html')
      self.data['tag'] = tag
      self.data['posts'] = posts
      self.data['title'] = "Tag: #{tag}"
    end
  end
end
